# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math
import sys
import torch
import torch.nn.functional as F

from fairseq import utils
from fairseq.models import transformer

from . import FairseqCriterion, register_criterion

def label_smoothed_nll_loss(
    samples, lprobs, cos_sim,
    epsilon, ignore_index=None, reduce=True):
    #similarity loss
    sim_loss = -torch.log((cos_sim+1)/2).mean(dim=0)

    cands = samples['cands'].view(-1, 1)
    cand_probs = -lprobs.gather(dim=-1,index=cands)

    if ignore_index is not None:
        pad_mask = cands.eq(ignore_index)
        cand_probs.masked_fill_(pad_mask, 0.)
    else:
        cand_probs = cand_probs.squeeze(-1)

    cand_loss = cand_probs.view(-1,samples['cands'].size(-1)).sum(dim=1) #[batch, seq_len] into [batch]
    cand_loss = cand_loss/(samples['cands'].ne(ignore_index).sum(dim=1).float())
    # the real len of each candidata
    cand_loss = cand_loss.mean(dim=0) #get the average for the batch

    total_loss = (cand_loss + sim_loss) * samples['ntokens']

    return total_loss, sim_loss, cand_loss

@register_criterion('label_smoothed_cross_entropy')
class LabelSmoothedCrossEntropyCriterion(FairseqCriterion):

    def __init__(self, args, task):
        super().__init__(args, task)
        self.eps = args.label_smoothing
        self.beam = args.cands_into_model
        self.update_freq = args.update_freq
        self.gpus = args.distributed_world_size
        self.use_batch_pearson = args.batch_pearson

    @staticmethod
    def add_args(parser):
        """Add criterion-specific arguments to the parser."""
        # fmt: off
        parser.add_argument('--label-smoothing', default=0., type=float, metavar='D',
                            help='epsilon for label smoothing, 0 means no label smoothing')
        parser.add_argument('--batch-pearson', default=False,action="store_true",
                            help='epsilon for label smoothing, 0 means no label smoothing')
        # fmt: on

    def forward(self, model, sample, train_step, reduce=True):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        src_hidden = model.decoder.through_ffnet(sample['src_bert_encoded'][:,0,:], flag='src')
        cand_hidden = model.decoder.through_ffnet(sample['cand_bert_encoded'][:,0,:], flag='cand')
        cand_net_logits = model.decoder.through_ffnet(sample['cand_bert_encoded'])

        total_loss, sim_loss, nll_loss = self.compute_loss(model, src_hidden, cand_hidden, cand_net_logits, sample)

        sample_size = sample['target'].size(0) if self.args.sentence_avg else sample['ntokens']
        logging_output = {
            #'total_loss': utils.item(total_loss.data) if reduce else total_loss.data,
            'loss': utils.item(total_loss.data) if reduce else total_loss.data,
            'sim_loss': utils.item(sim_loss.data) if reduce else sim_loss.data,
            'nll_loss': utils.item(nll_loss.data) if reduce else nll_loss.data,
            'ntokens': sample['ntokens'],
            'nsentences': sample['target'].size(0),
            'sample_size': sample_size,
        }
        del sample
        torch.cuda.empty_cache()
        return total_loss, sample_size, logging_output

    def compute_loss(self, model, src_hidden, cand_hidden, cand_net, sample, reduce=True):
        cand_lprobs = model.get_normalized_probs([cand_net], log_probs=True)
        cand_lprobs = cand_lprobs.view(-1, cand_lprobs.size(-1))

        cos_sim = F.cosine_similarity(src_hidden, cand_hidden, dim=1, eps=1e-6)

        bert_padding_idx = 0

        total_loss, sim_loss, nll_loss= label_smoothed_nll_loss(
            sample, cand_lprobs, cos_sim,
            self.eps, ignore_index=bert_padding_idx, reduce=reduce,
        )
        return total_loss, sim_loss, nll_loss

    @staticmethod
    def aggregate_logging_outputs(logging_outputs):
        """Aggregate logging outputs from data parallel training."""
        ntokens = sum(log.get('ntokens', 0) for log in logging_outputs)
        nsentences = sum(log.get('nsentences', 0) for log in logging_outputs)
        sample_size = sum(log.get('sample_size', 0) for log in logging_outputs)
        return {
            'loss': sum(log.get('loss', 0) for log in logging_outputs) / ntokens / math.log(2) if ntokens > 0 else 0.,
            'sim_loss': sum(log.get('sim_loss', 0) for log in logging_outputs) / len(logging_outputs),
            'nll_loss': sum(log.get('nll_loss', 0) for log in logging_outputs) / len(logging_outputs),
            'ntokens': ntokens,
            'nsentences': nsentences,
            'sample_size': sample_size,
        }
