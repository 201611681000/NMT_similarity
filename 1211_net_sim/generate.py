#!/usr/bin/env python3 -u
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
Translate pre-processed data with a trained model.
"""

import torch
import logging
import os
import sys

import math
import numpy as np
import sacrebleu
import random
import torch.nn.functional as F

from fairseq import bleu, checkpoint_utils, options, progress_bar, tasks, utils
from fairseq.data import data_utils
from fairseq.meters import StopwatchMeter, TimeMeter
from sacremoses import MosesTokenizer, MosesDetokenizer
from transformers import BertTokenizer, BertModel
#import bert_as_lm


def main(args):
    logging.basicConfig(
        format='%(asctime)s | %(levelname)s | %(name)s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        level=os.environ.get('LOGLEVEL', 'INFO').upper(),
        stream=sys.stdout,
    )
    logger = logging.getLogger('fairseq_cli.generate')

    assert args.path is not None, '--path required for generation!'
    assert not args.sampling or args.nbest == args.beam, \
        '--sampling requires --nbest to be equal to --beam'
    assert args.replace_unk is None or args.raw_text, \
        '--replace-unk requires a raw text dataset (--raw-text)'

    utils.import_user_module(args)

    if args.max_tokens is None and args.max_sentences is None:
        args.max_tokens = 12000
    #print(args)

    use_cuda = torch.cuda.is_available() and not args.cpu

    # Load dataset splits
    task = tasks.setup_task(args)
    task.load_dataset(args.gen_subset)

    # Set dictionaries
    try:
        src_dict = getattr(task, 'source_dictionary', None)
    except NotImplementedError:
        src_dict = None
    tgt_dict = task.target_dictionary

    # Load ensemble
    logger.info('loading model(s) from {}'.format(args.path))
    models, _model_args = checkpoint_utils.load_model_ensemble(
        args.path.split(':'),
        arg_overrides=eval(args.model_overrides),
        task=task,
    )

    mose_en = MosesDetokenizer(lang='en')
    mose_de = MosesDetokenizer(lang='de')
    if args.bert_model_path :
        bert_tokenizer = BertTokenizer.from_pretrained(args.bert_model_path)
        bert_model = BertModel.from_pretrained(args.bert_model_path)
        bert_model.cuda()
        bert_model.eval()

        bert_de_tokenizer = BertTokenizer.from_pretrained(args.bert_german_path)
        bert_de_model = BertModel.from_pretrained(args.bert_german_path)
        bert_de_model.cuda()
        bert_de_model.eval()
    # Optimize ensemble for generation
    for model in models:
        model.make_generation_fast_(
            beamable_mm_beam_size=None if args.no_beamable_mm else args.beam,
            need_attn=args.print_alignment,
        )
        model.eval()
        if args.fp16:
            model.half()
        if use_cuda:
            model.cuda()

    # Load alignment dictionary for unknown word replacement
    # (None if no unknown word replacement, empty if no path to align dictionary)
    align_dict = utils.load_align_dict(args.replace_unk)

    # Load dataset (possibly sharded)
    itr = task.get_batch_iterator(
        dataset=task.dataset(args.gen_subset),
        max_tokens=args.max_tokens,
        max_sentences=args.max_sentences,
        max_positions=utils.resolve_max_positions(
            task.max_positions(),
            *[model.max_positions() for model in models]
        ),
        ignore_invalid_inputs=args.skip_invalid_size_inputs_valid_test,
        required_batch_size_multiple=args.required_batch_size_multiple,
        num_shards=args.num_shards,
        shard_id=args.shard_id,
        num_workers=args.num_workers,
    ).next_epoch_itr(shuffle=False)
    batch_len = len(task.dataset(args.gen_subset))

    # Initialize generator
    gen_timer = StopwatchMeter()
    generator = task.build_generator(args)

    #Bert model
    #bert_model = bert_as_lm.Bert_score(torch.cuda.current_device()) 
    #mose_de = MosesDetokenizer(lang='en')
    # Generate and compute BLEU score
    if args.sacrebleu:
        scorer = bleu.SacrebleuScorer()
    else:
        scorer = bleu.Scorer(tgt_dict.pad(), tgt_dict.eos(), tgt_dict.unk())
    num_sentences = 0
    has_target = True

    total_pearson = []
    total_bert_pearson = []
    random_list = []
    bert_bleu_equal = 0
    sents_num = 0
    if args.gen_subset == 'train':
        random_list = [i for i in range(0, batch_len)]
        random.shuffle(random_list)
        random_list = random_list[:1000]

    with progress_bar.build_progress_bar(args, itr) as t:
        wps_meter = TimeMeter()
        for sample in t:
            selected = []
            if random_list:
                for i in sample['id']:
                    selected.append(True) if i in random_list else selected.append(False)
                selected = torch.nonzero(torch.tensor(selected).ne(0)).squeeze(-1)
                if len(selected) == 0:
                    continue
                for item in sample.keys():
                    if item == 'nsentences' or item== 'ntokens':
                        continue
                    elif item == 'net_input':
                        for input in sample[item].keys():
                            sample[item][input] = sample[item][input].index_select(0,selected)
                    else: sample[item] = sample[item].index_select(0,selected)
                sample['nsentences'] = len(selected)
                sample['ntokens'] = torch.LongTensor([s.ne(1).long().sum() for s in sample['target']]).sum().item()

            sample = utils.move_to_cuda(sample) if use_cuda else sample
            if 'net_input' not in sample:
                continue

            prefix_tokens = None
            if args.prefix_size > 0:
                prefix_tokens = sample['target'][:, :args.prefix_size]

            gen_timer.start()
            hypos = task.inference_step(generator, models, sample, prefix_tokens)
            num_generated_tokens = sum(len(h[0]['tokens']) for h in hypos)
            gen_timer.stop(num_generated_tokens)

            for i, sample_id in enumerate(sample['id'].tolist()):
                sents_num += 1
                if random_list and sample_id not in random_list:
                    continue
                has_target = sample['target'] is not None

                # Remove padding
                src_tokens = utils.strip_pad(sample['net_input']['src_tokens'][i, :], tgt_dict.pad())
                target_tokens = None
                if has_target:
                    target_tokens = utils.strip_pad(sample['target'][i, :], tgt_dict.pad()).int().cpu()

                # Either retrieve the original sentences or regenerate them from tokens.
                if align_dict is not None:
                    src_str = task.dataset(args.gen_subset).src.get_original_text(sample_id)
                    target_str = task.dataset(args.gen_subset).tgt.get_original_text(sample_id)
                else:
                    if src_dict is not None:
                        src_str = src_dict.string(src_tokens, args.remove_bpe)
                        src_splittokens = src_str.split(' ')
                        src_str_for_bert = mose_de.detokenize(src_splittokens)
                        src_ids, src_bert = \
                                get_bert_out(src_str_for_bert, bert_de_tokenizer, bert_de_model)
                    else:
                        src_str = ""
                    if has_target:
                        target_str = tgt_dict.string(target_tokens, args.remove_bpe, escape_unk=True)

                if not args.quiet:
                    print ("---------------{}--------------".format(sents_num))
                    if src_dict is not None:
                        print('S-{}\t{}'.format(sample_id, src_str), file=sys.stdout)
                    if has_target:
                        print('T-{}\t{}'.format(sample_id, target_str), file=sys.stdout)

                # Process top predictions
                probs = []
                bleu_score = []
                cands = []
                sents_bert_score = []
                detoken_cands =[]
                temp_cand_tokens = []

                for j, hypo in enumerate(hypos[i][:args.nbest]):
                    hypo_tokens, hypo_str, alignment = utils.post_process_prediction(
                        hypo_tokens=hypo['tokens'].int().cpu(),
                        src_str=src_str,
                        alignment=hypo['alignment'],
                        align_dict=align_dict,
                        tgt_dict=tgt_dict,
                        remove_bpe=args.remove_bpe,
                    )

                    if not args.quiet:
                        score = hypo['score'] / math.log(2)  # convert to base 2
                    probs.append(score)
                    single_score = sacrebleu.corpus_bleu([hypo_str],[[target_str]], use_effective_order=True, tokenize="none")
                    bleu_score.append(single_score.score)
                    cands.append(hypo_str)
                    temp_cand_tokens.append(hypo['tokens'].cpu())

                    hypo_splittokens = hypo_str.split(' ')
                    detoken_hypo = mose_de.detokenize(hypo_splittokens)
                    detoken_cands.append(detoken_hypo)

                #print (cands)
                cand_ids, cand_bert = get_bert_out(detoken_cands, bert_tokenizer, bert_model)
                src_hidden = models[0].decoder.through_ffnet(src_bert[:,0,:],'src')
                cand_hidden = models[0].decoder.through_ffnet(cand_bert[:,0,:],'cand')
                cand_net_vocab = models[0].decoder.through_ffnet(cand_bert)
                cand_lprob = \
                        models[0].get_normalized_probs([cand_net_vocab],log_probs=True).view(-1,cand_net_vocab.size(-1))
                #compute similarity
                cos_sim = F.cosine_similarity(
                    src_hidden.repeat(args.nbest,1), cand_hidden, dim=1, eps=1e-6) #[beam]
                cos_sim = torch.log((cos_sim + 1)/2)
                cand_probs = cand_lprob.gather(dim=-1,index=cand_ids.view(-1,1))
                cand_loss = cand_probs.view(-1,cand_ids.size(-1)).sum(dim=1)
                cand_loss = cand_loss/cand_ids.ne(0).sum(dim=1).float()
                #print (cand_loss)
                total_score = cos_sim + 0.1*cand_loss
                #print (total_score)
                net_pos = torch.argmax(total_score).item()

                pearson = 0
                #if args.nbest > 20:
                np_prob = np.array(probs)
                np_bleu = np.array(bleu_score)
                pearson = np.corrcoef(np_prob, np_bleu)[0][1]

                if not np.isnan(pearson):
                    total_pearson.append(pearson)
                #else:
                #    print ("cands:", cands, file=sys.stdout)
                #    print ("probs:", np_prob, file=sys.stdout)
                #    print ("bleus:", np_bleu, file=sys.stdout)

                bleu_pos = bleu_score.index(max(bleu_score))
                print ("-----bleu choice: {} bleu:{:.3f}  pos: {}".format(
                            cands[bleu_pos], bleu_score[bleu_pos], bleu_pos+1),file=sys.stdout)
                pos = probs.index(max(probs))
                print ("-----prob choice: {} bleu:{:.3f} pos: {}".format(
                        cands[pos], bleu_score[pos],pos+1),file=sys.stdout)
                print ("-----net choice: {} bleu:{:.3f} pos: {} score:{:.3f}".format(
                        cands[net_pos], bleu_score[net_pos],net_pos+1, total_score[net_pos]),file=sys.stdout)

                '''
                np_bert = np.array(sents_bert_score)
                bert_bleu_pearson = np.corrcoef(np_bert, np_bleu)[0][1]
                if not np.isnan(bert_bleu_pearson):
                   total_bert_pearson.append(bert_bleu_pearson) 

                bert_pos = sents_bert_score.index(min(sents_bert_score))
                print('*****{} bert choice: {}\tprob:{:.3f}\tbleu:{:.3f}\tbertscore:{:.3f}\tposition:{}\tprob_bleu_pearson:{:.3f} bert_bleu_p: {:.3f} '. \
                format(sample_id, cands[bert_pos], probs[bert_pos], bleu_score[bert_pos],
                    sents_bert_score[bert_pos], bert_pos+1, pearson, bert_bleu_pearson), file=sys.stdout)
                '''
                if args.usebleu:
                    final_hypo = cands[bleu_pos]
                elif args.usebert:
                    final_hypo = cands[net_pos]
                else: final_hypo = cands[pos]
                scorer.add_string(target_str, final_hypo)

                print ('H choice use bleu: {} usebert: {}'.format(args.usebleu, args.usebert))
                if has_target and sents_num % 800 == 0:
                    print('Generate {} with beam={}: {}\t{}'.format(args.gen_subset, args.beam,
                                                                    scorer.result_string(),
                                                                    sents_num,file=sys.stdout))
            wps_meter.update(num_generated_tokens)
            #t.log({'wps': round(wps_meter.avg)})
            num_sentences += sample['nsentences']

    logger.info('Translated {} sentences ({} tokens) in {:.1f}s ({:.2f} sentences/s, {:.2f} tokens/s)'.format(
        num_sentences, gen_timer.n, gen_timer.sum, num_sentences / gen_timer.sum, 1. / gen_timer.avg))
    if has_target:
        print('Generate {} with beam={}: {}\n --- prob&bleu pearson: {:.4f} ---'.format(
            args.gen_subset, args.beam, scorer.result_string(), 
            sum(total_pearson)/len(total_pearson)), file=sys.stdout)

    return scorer

def get_decoded_out(src_tokens, src_len, cands_token, models):
    beam = len(cands_token)
    pad_idx = 1
    eos_idx = 2
    left_pad_target = False
    src_tokens = src_tokens.repeat(beam,1)
    src_len = src_len.repeat(1,beam)
    prev_out = data_utils.collate_tokens(
        cands_token,
        pad_idx,
        eos_idx,
        left_pad = left_pad_target,
        move_eos_to_beginning=True,).cuda()
    out = models[0](src_tokens, src_len, prev_out, features_only=True)[0]
    out = out.mean(dim=1)
    return out

def get_bert_out(string, token, model):
    token_input = token(string, return_tensors='pt',padding=True)
    for key in token_input:
        token_input[key] = token_input[key].cuda()
    bert_output = model(**token_input)[0]
    return token_input['input_ids'],bert_output

def cli_main():
    parser = options.get_generation_parser()
    args = options.parse_args_and_arch(parser)
    main(args)


if __name__ == '__main__':
    cli_main()
