import random
import torch
import numpy as np
UNK, SEP, PAD, CLS, MASK = "[UNK]", "[SEP]", "[PAD]", "[CLS]", "[MASK]"

def load_label_dict(label_path):
    label_dict, id2label_dict = {}, {}
    with open(label_path, 'r', encoding = 'utf8') as i:
        lines = i.readlines()
        for l in lines:
            content_list = l.strip('\n').split()
            label = content_list[0]
            label_id = int(content_list[1])
            label_dict[label] = label_id
            id2label_dict[label_id] = label
    return label_dict, id2label_dict

class Data:
    def __init__(self, tokenizer, train_path, dev_path, test_path, label_path, max_len):
        self.train_path, self.dev_path, self.test_path, self.label_path = \
        train_path, dev_path, test_path, label_path
        self.label_dict, self.id2label_dict = load_label_dict(label_path)
        self.num_class = len(self.label_dict)
        print ('number of tags is {}'.format(self.num_class))
        self.max_len = max_len

        self.tokenizer = tokenizer
        self.unk_idx, self.sep_idx, self.pad_idx, self.cls_idx, self.mask_idx = \
        self.tokenizer.convert_tokens_to_ids([UNK, SEP, PAD, CLS, MASK])

        self.train_token_id_list, self.train_tag_id_list = self.process_file(train_path)
        self.dev_token_id_list, self.dev_tag_id_list = self.process_file(dev_path)
        self.test_token_id_list, self.test_tag_id_list = self.process_file(test_path)

        self.train_num, self.dev_num, self.test_num = len(self.train_token_id_list), \
        len(self.dev_token_id_list), len(self.test_token_id_list)
        print ('training number is {}, dev number is {}, test_num is {}'.format(self.train_num, 
            self.dev_num, self.test_num))
        self.train_idx_list = [i for i in range(self.train_num)]
        random.shuffle(self.train_idx_list)
        self.dev_idx_list = [j for j in range(self.dev_num)]
        self.test_idx_list = [j for j in range(self.test_num)]
        self.dev_current_idx, self.test_current_idx = 0, 0

        max_train_seq_len = 0
        for item in self.train_token_id_list:
            max_train_seq_len = max(len(item), max_train_seq_len)
        max_dev_seq_len = 0
        for item in self.dev_token_id_list:
            max_dev_seq_len = max(len(item), max_dev_seq_len)
        max_test_seq_len = 0
        for item in self.test_token_id_list:
            max_test_seq_len = max(len(item), max_test_seq_len)
        print ('Maximum train sequence length: %d, dev sequence length: %d, test sequence length: %d' % \
            (max_train_seq_len, max_dev_seq_len, max_test_seq_len))

    def process_instance(self, line):
        content_list = line.strip('\n').split('\t')
        assert len(content_list) == 2
        token_list, tag_name_list = content_list[0].split(), content_list[1].split()
        token_list = token_list[:self.max_len]
        tag_name_list = tag_name_list[:self.max_len]
        assert len(token_list) == len(tag_name_list)
        token_list = [CLS] + token_list + [SEP]
        tag_name_list = ['O'] + tag_name_list + ['O']
        token_id_list = self.tokenizer.convert_tokens_to_ids(token_list)
        tag_list = [self.label_dict[token] for token in tag_name_list]
        assert len(token_id_list) == len(tag_list)
        return token_id_list, tag_list

    def process_file(self, path):
        all_token_id, all_tag_id = [], []
        with open(path, 'r', encoding = 'utf8') as i:
            lines = i.readlines()
            for l in lines:
                one_token_id_list, one_tag_id_list = self.process_instance(l)
                all_token_id.append(one_token_id_list)
                all_tag_id.append(one_tag_id_list)
        return all_token_id, all_tag_id

    def process_input(self, batch_inp):
        max_len = max([len(item) for item in batch_inp])
        xs = []
        for item in batch_inp:
            x = item + [self.pad_idx]*(max_len - len(item))
            xs.append(x)
        src_tensor = torch.LongTensor(xs).contiguous()
        attn_mask = ~src_tensor.eq(self.pad_idx)
        return src_tensor, attn_mask

    def process_output(self, batch_out):
        o_tag_id = self.label_dict['O']
        max_len = max([len(item) for item in batch_out])
        ys, masks = [], []
        for item in batch_out:
            y = item + [o_tag_id]*(max_len - len(item))
            msk = [1.0]*len(item) + [0.0]*(max_len - len(item))
            ys.append(y)
            masks.append(msk)
        tgt_tensor = torch.LongTensor(ys).contiguous()
        tgt_mask = torch.tensor(masks, dtype=torch.uint8).contiguous()
        return tgt_tensor, tgt_mask

    def process_batch_data(self, batch_inp, batch_out):
        src_tensor, src_attn_mask = self.process_input(batch_inp)
        tgt_tensor, tgt_mask = self.process_output(batch_out)
        return src_tensor, src_attn_mask, tgt_tensor, tgt_mask

    def get_next_train_batch(self, batch_size):
        batch_idx_list = random.sample(self.train_idx_list, batch_size)
        batch_token_id_list, batch_tag_id_list = [], []
        for idx in batch_idx_list:
            batch_token_id_list.append(self.train_token_id_list[idx])
            batch_tag_id_list.append(self.train_tag_id_list[idx])
        src_tensor, src_attn_mask, tgt_tensor, tgt_mask = \
        self.process_batch_data(batch_token_id_list, batch_tag_id_list)
        return src_tensor, src_attn_mask, tgt_tensor, tgt_mask

    def get_next_validation_batch(self, batch_size, mode):
        batch_token_id_list, batch_tag_id_list, batch_ref_id_list = [], [], []
        if mode == 'dev':
            curr_select_idx, instance_num = self.dev_current_idx, self.dev_num
            token_id_list, tag_id_list = self.dev_token_id_list, self.dev_tag_id_list
        elif mode == 'test':
            curr_select_idx, instance_num = self.test_current_idx, self.test_num
            token_id_list, tag_id_list = self.test_token_id_list, self.test_tag_id_list
        else:
            raise Exception('Wrong Validation Mode!!!')

        if curr_select_idx + batch_size < instance_num:
            for i in range(batch_size):
                curr_idx = curr_select_idx + i
                batch_token_id_list.append(token_id_list[curr_idx])
                batch_tag_id_list.append(tag_id_list[curr_idx])
                batch_ref_id_list.append(tag_id_list[curr_idx])
            if mode == 'dev':
                self.dev_current_idx += batch_size
            else:
                self.test_current_idx += batch_size
        else:
            for i in range(batch_size):
                curr_idx = curr_select_idx + i
                if curr_idx > instance_num - 1: # 对dev_current_idx重新赋值
                    curr_idx = 0
                    if mode == 'dev':
                        self.dev_current_idx = 0
                    else:
                        self.test_current_idx = 0
                batch_token_id_list.append(token_id_list[curr_idx])
                batch_tag_id_list.append(tag_id_list[curr_idx])
                batch_ref_id_list.append(tag_id_list[curr_idx])
            if mode == 'dev':
                self.dev_current_idx = 0
            else:
                self.test_current_idx = 0
        src_tensor, src_attn_mask, tgt_tensor, tgt_mask = \
        self.process_batch_data(batch_token_id_list, batch_tag_id_list)
        return src_tensor, src_attn_mask, tgt_tensor, tgt_mask, batch_ref_id_list

    def convert_tag_id_to_name(self, tag_id_list):
        return [self.id2label_dict[idx] for idx in tag_id_list]

    def parse_result(self, decode_list):
        res = []
        for y in decode_list:
            res.append(' '.join(self.convert_tag_id_to_name(y[1:-1]))) # remove CLS and SEP
        return res

