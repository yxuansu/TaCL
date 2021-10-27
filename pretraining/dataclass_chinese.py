import random
import torch
import numpy as np
import re
from google_bert import create_instances_from_document

UNK, SEP, PAD, CLS, MASK = "[UNK]", "[SEP]", "[PAD]", "[CLS]", "[MASK]"
class PretrainCorpus(object):
    def __init__(self, tokenizer, filename, max_len, whole_word_masking=False):
        '''
            tokenizer: BERT tokenizer
            filename: pretraining corpus
            max_len: maximum length for each sentence
        '''
        self.tokenizer = tokenizer
        self.vocab_size = self.tokenizer.vocab_size
        self.special_token_id_list = tokenizer.convert_tokens_to_ids([UNK, SEP, PAD, CLS, MASK])
        self.unk_id, self.sep_id, self.pad_id, self.cls_id, self.mask_id = self.special_token_id_list
        self.max_len = max_len
        self.filename = filename
        self.stream = open(self.filename, encoding='utf8')
        self.epoch_id = 0
        self.whole_word_masking = whole_word_masking
        self.load_lines(filename)

    def load_lines(self, filename):
        with open(filename, 'r', encoding = 'utf8') as i:
            lines = i.readlines()
        print ('Number of lines is {}'.format(len(lines)))
        docs = [[]]
        for line in lines:
            tokens = line.strip('\n').strip().split()
            if tokens:
                docs[-1].append(tokens)
            else:
                docs.append([])

        docs = [x for x in docs if x] # filter out empty lines
        self.docs = docs
        random.shuffle(docs)

        data = []
        for idx, doc in enumerate(docs):
            data.extend(create_instances_from_document(docs, idx, self.max_len))
        self.data = data
        print ('number of sentence pairs is {}'.format(len(self.data)))
        self.train_idx_list = [i for i in range(len(self.data))]

    def get_batch_data(self, batch_size):
        batch_idx_list = random.sample(self.train_idx_list, batch_size)
        batch_data = []
        for idx in batch_idx_list:
            batch_data.append(self.data[idx])
        return self.batchify(batch_data)

    def random_token(self):
        rand_idx = 1 + np.random.randint(self.vocab_size-1)
        while rand_idx in self.special_token_id_list:
            rand_idx = 1 + np.random.randint(self.vocab_size-1)
        random_token = self.tokenizer.convert_ids_to_tokens([rand_idx])[0]
        return random_token

    def random_mask(self, tokens, masked_lm_prob, max_predictions_per_seq):
        num_to_predict = min(max_predictions_per_seq, max(1, int(round(len(tokens) * masked_lm_prob))))
        masked_tokens, mask = [], []
        cand = []
        for i, token in enumerate(tokens):
            #if token == CLS or token == SEP:
            if token in [UNK, SEP, PAD, CLS, MASK]: # do not learn to predict special tokens
                continue
            cand.append(i)
        random.shuffle(cand)
        cand = set(cand[:num_to_predict])

        masked_tokens, mask, tgt = [], [], []
        for i, token in enumerate(tokens):
            if i in cand:
                if random.random() < 0.8:
                    masked_tokens.append(MASK)
                else:
                    if random.random() < 0.5:
                        masked_tokens.append(token)
                    else:
                        masked_tokens.append(self.random_token())
                mask.append(1)
                tgt.append(token)
            else:
                masked_tokens.append(token)
                mask.append(0)
                tgt.append(PAD)
        return masked_tokens, mask, tgt

    def whole_word_random_mask(self, tokens, masked_lm_prob):
        # tokens is a list of words, each work could contain multiple characters
        num_tokens = len(tokens)
        num_to_predict = min(int(num_tokens*0.25), max(1, int(round(len(tokens) * masked_lm_prob))))
        masked_tokens, mask = [], []
        cand = [] # token index 
        total_char_num = 0
        for i, token in enumerate(tokens):
            #if token == CLS or token == SEP:
            if token in [UNK, SEP, PAD, CLS, MASK]: # do not learn to predict special tokens
                continue
            cand.append(i) 
        random.shuffle(cand)
        cand = set(cand[:num_to_predict])

        masked_tokens, mask, tgt = [], [], []
        for i, token in enumerate(tokens):
            if i in cand:
                for char in token:
                    if random.random() < 0.8:
                        masked_tokens.append(MASK)
                    else:
                        if random.random() < 0.5:
                            masked_tokens.append(char)
                        else:
                            masked_tokens.append(self.random_token())
                    mask.append(1)
                    tgt.append(char)
            else:
                if token in [UNK, SEP, PAD, CLS, MASK]: # special tokens
                    masked_tokens.append(token)
                    mask.append(0)
                    tgt.append(PAD)
                else:
                    for char in token:
                        masked_tokens.append(char)
                        mask.append(0)
                        tgt.append(PAD)
        masked_tokens = masked_tokens[:-1][:self.max_len-1] + [masked_tokens[-1]]
        mask = mask[:-1][:self.max_len-1] + [mask[-1]] # keep number of max_len characters
        tgt = tgt[:-1][:self.max_len-1] + [tgt[-1]]
        assert len(masked_tokens) == len(mask)
        assert len(mask) == len(tgt)
        return masked_tokens, mask, tgt

    def ListsToTensor(self, xs, tokenize=False):
        max_len = max(len(x) for x in xs)
        ys = []
        for x in xs:
            if tokenize:
                y = self.tokenizer.convert_tokens_to_ids(x) + [self.pad_id]*(self.max_len - len(x))
            else:
                y = x + [0]*(self.max_len -len(x))
            ys.append(y)
        data = torch.LongTensor(ys).contiguous() # bsz x seqlen
        return data # bsz x seqlen

    def process_tgt(self, tgt_matrix):
        max_len = max(len(x) for x in tgt_matrix)
        ys = []
        for x in tgt_matrix:
            y = self.tokenizer.convert_tokens_to_ids(x) + [self.pad_id]*(self.max_len - len(x))
            ys.append(y)
        ys = torch.LongTensor(ys).contiguous() # bsz x seqlen
        labels = ys.clone()
        labels[labels[:, :] == self.pad_id] = -100
        contrastive_labels = ys.clone()
        contrastive_labels[contrastive_labels[:, :] == self.pad_id] = 0
        contrastive_labels[contrastive_labels[:, :] != self.pad_id] = 1
        return labels, contrastive_labels.type(torch.FloatTensor)

    def batchify(self, data):
        truth, inp, seg, msk, tgt_matrix = [], [], [], [], []
        nxt_snt_flag = []
        for a, b, r in data:
            #print (a, b, r)
            #raise Exception()
            '''
                a, b are two tokenized sequence
                r indicates whether two sentences are continuation, True or False
            '''
            
            if self.whole_word_masking:
                x = [CLS]+a+[SEP]+b+[SEP]
                masked_x, mask, tgt = self.whole_word_random_mask(x, 0.15)
                truth_char = [CLS]
                seg_char = []
                for word in a:
                    for char in word:
                        truth_char.append(char)
                truth_char.append(SEP)
                seg_char += [0 for _ in truth_char]
                for word in b:
                    for char in word:
                        truth_char.append(char)
                        seg_char.append(1)
                truth_char.append(SEP)
                seg_char.append(1)
                assert len(truth_char) == len(seg_char)
                truth_char = truth_char[:-1][:self.max_len-1] + [truth_char[-1]]
                seg_char = seg_char[:-1][:self.max_len-1] + [seg_char[-1]]
                truth.append(truth_char)
                seg.append(seg_char)
                assert len(masked_x) == len(mask)
                assert len(mask) == len(truth_char)
                assert len(truth_char) == len(seg_char)
            else:
                x = [CLS]+a+[SEP]+b+[SEP]
                truth.append(x)
                seg.append([0]*(len(a)+2) + [1]*(len(b)+1))
                masked_x, mask, tgt = self.random_mask(x, 0.15, self.max_len * 0.25)

            inp.append(masked_x)
            msk.append(mask)
            tgt_matrix.append(tgt)
            if r: # r stands for is_random_text
                nxt_snt_flag.append(1)
            else:
                nxt_snt_flag.append(0)

        truth = self.ListsToTensor(truth, tokenize=True)
        inp = self.ListsToTensor(inp, tokenize=True)
        seg = self.ListsToTensor(seg, tokenize=False)
        msk = self.ListsToTensor(msk, tokenize=False).to(torch.uint8) # bsz x seqlen
        attn_msk = ~inp.eq(self.pad_id)
        nxt_snt_flag = torch.ByteTensor(nxt_snt_flag)
        labels, contrastive_labels = self.process_tgt(tgt_matrix)
        return truth, inp, seg, msk, attn_msk, labels, contrastive_labels, nxt_snt_flag
