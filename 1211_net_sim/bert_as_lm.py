import sys
import torch
import math
import argparse
import datetime

import torch.nn.functional as F
from torch.nn import CrossEntropyLoss, MSELoss
from transformers import BertTokenizer, BertForMaskedLM


class InputFeatures(object):
    def __init__(self, input_ids, input_labels, segment_ids, input_mask, masked_lm_positions,
               masked_lm_ids):
        self.input_ids = input_ids
        self.orig_input_labels = input_labels
        self.segment_ids = segment_ids
        self.input_mask = input_mask
        self.masked_lm_positions = masked_lm_positions
        self.masked_lm_ids = masked_lm_ids

def is_subtoken(token):
    return token.startswith("##")

class Bert_score(object):
    def __init__(self, cuda_index):

        self.cuda_index = abs(cuda_index - 1)
        MODEL_PATH = '/ceph_data/intn/solarawang/bert_model/'
        MODEL_PATH = '/ceph_int/solarawang/bert_model/'
        self.tokenizer = BertTokenizer.from_pretrained(MODEL_PATH)
        self.model = BertForMaskedLM.from_pretrained(MODEL_PATH, return_dict=True)
        #self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        #self.model = BertForMaskedLM.from_pretrained('bert-base-uncased', return_dict=True)
        self.device = torch.device("cuda:"+str(self.cuda_index) if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        self.MASK_token = '[MASK]'
        self.MASK_ID = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(self.MASK_token))[0]
        self.model.eval()

        self.starttime = datetime.datetime.now()

    def process_examples (self, data_list):
        input_features = self.tokenizer(data_list, return_tensors="pt", padding=True)
        mask_sent_features = []
        for i,sent in enumerate(data_list):
            input_tokens = self.tokenizer.tokenize(sent)
            sent_features = self.mask_single_sent(i, input_tokens, input_features)
            mask_sent_features.append(sent_features)
        #endtime = datetime.datetime.now()
        #print ("after process_examples", (endtime-self.starttime))
        return mask_sent_features


    def mask_single_sent (self, sent_index, input_tokens, features):
        #input_ids = features['input_ids'][sent_index].tolist()
        #segment_ids = features['token_type_ids'][sent_index].tolist()
        #input_mask = features['attention_mask'][sent_index].tolist()

        input_ids = features['input_ids'][sent_index]
        segment_ids = features['token_type_ids'][sent_index]
        input_mask = features['attention_mask'][sent_index]

        input_tokens.insert(0,'[CLS]')
        input_tokens.append('[SEP]')

        index = 1
        features = []
        while index < len(input_tokens) - 1: #get rid of [CLS] and [SEP] and the final mark
            mask_num = 1
            while is_subtoken(input_tokens[index+mask_num]):
                mask_num += 1

            new_mask_input_ids, masked_tokenid, masked_pos = self.mask_one_word(input_ids, mask_num, index)
            index += mask_num

            feature = InputFeatures(
                input_ids = new_mask_input_ids,
                input_labels = input_ids,
                segment_ids = segment_ids,
                input_mask = input_mask,
                masked_lm_positions = masked_pos,
                masked_lm_ids = masked_tokenid,
            )
            features.append(feature)
            #endtime = datetime.datetime.now()
            #print ("after mask one sentence:", (endtime-self.starttime))
        return features

    def mask_one_word (self, input_ids, mask_num, word_index):
        new_mask_input_ids = input_ids.clone()
        mask_tokenid = []
        mask_pos = list(range(word_index, word_index+mask_num))
        masked_pos_list = [0 for i in range (len(input_ids))]
        for i in mask_pos:
            masked_pos_list[i] = 1
            new_mask_input_ids[i] = self.MASK_ID
            mask_tokenid.append(input_ids[i])
        return new_mask_input_ids, mask_tokenid, masked_pos_list

    def get_model_output (self, features):
        all_input_ids = []
        all_input_labels = []
        all_input_mask = []
        all_segment_ids = []
        all_masked_pos = []
        all_masked_tokenids = []
        sentences_masked_counts = []

        for sent_feature in features:
            mask_count = 0
            for feature in sent_feature:
                mask_count += 1
                all_input_ids.append(feature.input_ids)
                all_input_labels.append(feature.orig_input_labels)
                all_input_mask.append(feature.input_mask)
                all_segment_ids.append(feature.segment_ids)
                all_masked_pos.append(feature.masked_lm_positions)
                all_masked_tokenids.append(feature.masked_lm_ids)
            sentences_masked_counts.append(mask_count)

        all_input_ids = torch.stack(all_input_ids).cuda(self.cuda_index)
        all_input_mask = torch.stack(all_input_mask).cuda(self.cuda_index)
        all_segment_ids = torch.stack(all_segment_ids).cuda(self.cuda_index)
        all_input_labels = torch.stack(all_input_labels)
        #all_masked_pos = torch.stack(all_masked_pos)
        #all_masked_tokenids = torch.stack(all_masked_tokenids)

        #need to turn all the variables into tensor!
        torch.cuda.empty_cache()
        with torch.no_grad():
            output = self.model(input_ids=all_input_ids,
              attention_mask=all_input_mask,token_type_ids=all_segment_ids)
        #endtime = datetime.datetime.now()
        #print ("get the model output:", (endtime-self.starttime))
        del all_input_ids, all_input_mask, all_segment_ids
        return output, all_input_labels, all_masked_pos, sentences_masked_counts

    def get_masked_word_score (self, input_labels,input_features, masked_pos, sents_masked_counts, output):
        masked_pos = torch.Tensor(masked_pos)
        scores = F.softmax(output.logits,dim=-1).cpu()
        #loss = output.loss
        #scores = output.logits
        input_labels = input_labels.unsqueeze(0).view(input_labels.size(0),-1,1)
        labels_score = scores.gather(dim=-1, index = input_labels)
        labels_prob = torch.log(labels_score.view(labels_score.size(0),-1)).cpu()

        masked_prob = labels_prob.mul(masked_pos).sum(1)

        prev = 0
        sents_score = []
        for count in sents_masked_counts:
            sent_score = masked_prob[prev:prev+count]
            sent_score = sent_score.sum().item()
            sent_ppl = math.pow(2,-sent_score/count)
            sents_score.append(sent_ppl)
            prev = prev + count

        torch.cuda.empty_cache()

        #endtime = datetime.datetime.now()
        #print ("process masked output:", (endtime-self.starttime))
        return sents_score

    def bert_score (self, candidates):
        input_features = self.process_examples(candidates)
        output, input_labels, masked_pos, sent_masked_counts = self.get_model_output(input_features)
        return self.get_masked_word_score(input_labels, input_features, masked_pos, sent_masked_counts, output)

#candidates = ['he had just heard a demonstration of beethoven &apos;s first and fourth symphony , and he came out behind the stage to imagine .',
#              'he had just heard a demonstration of beethoven &apos;s first and fourth symphony and came behind the stage to imagine me .',
#              'he had just heard a demonstration of beethoven &apos;s first and fourth symphony , and he came out behind the stage to imagine .',
#              ]
#candidates = ['there is a book on the desk .']
#temp = Bert_score()
#sents_score = temp.bert_score(candidates)
#print (candidates)
#print (sents_score)
