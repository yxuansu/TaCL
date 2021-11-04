import torch
def text_to_id(text, tokenizer, is_cuda, device):
    text = '[CLS] ' + text + ' [SEP]'
    tokens = tokenizer.tokenize(text)
    tokens_id = tokenizer.convert_tokens_to_ids(tokens)
    tokens_id = torch.LongTensor(tokens_id).view(1,-1)
    if is_cuda:
        tokens_id = tokens_id.cuda(device)
    return tokens_id

import numpy as np
def get_diag_mask_matrix(seqlen):
    a = np.zeros((seqlen, seqlen), float)
    np.fill_diagonal(a, 1.0)
    return 1.0 - a

import torch.nn.functional as F
def compute_cosine_correlation_matrix(hidden):
    # hidden: 1 x seqlen x embed_dim
    _, seq_len, _ = hidden.size()
    norm_hidden = hidden / hidden.norm(dim=2, keepdim=True) # normalize vectors to unit norm
    correlation_matrix = torch.matmul(norm_hidden, norm_hidden.transpose(1,2)).view(seq_len, seq_len)
    correlation_matrix = correlation_matrix.detach().cpu().numpy()
    return correlation_matrix

def compute_stat(hidden):
    cosine_correlation_matrix = compute_cosine_correlation_matrix(hidden)
    seq_len = cosine_correlation_matrix.shape[0]
    mask_matrix = get_diag_mask_matrix(seq_len)
    assert mask_matrix.shape == cosine_correlation_matrix.shape
    cosine_masked_matrix = cosine_correlation_matrix * mask_matrix # mask out the diagonal elements
    sum_cosine = np.sum(cosine_masked_matrix) / 2
    element_num = seq_len * (seq_len - 1) / 2
    return sum_cosine, element_num

def process_one_instance(model, tokenizer, text, is_cuda, device):
    res_dict = {}
    for idx in range(1,13):
        res_dict[idx] = {}

    input_id = text_to_id(text, tokenizer, is_cuda, device)
    outputs = model(input_id, output_hidden_states=True)
    attention_hidden_states = outputs[-1]
    for idx in range(1,13):
        one_cosine_sum, one_token_num = compute_stat(attention_hidden_states[idx])
        res_dict[idx]['cosine_sum'] = one_cosine_sum
        res_dict[idx]['token_sum'] = one_token_num
    return res_dict

from transformers import AutoModel, AutoTokenizer
def load_model_tokenizer(model_name):
    model = AutoModel.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return model, tokenizer

import argparse
def parse_config():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str)
    parser.add_argument("--file_path", type=str)
    parser.add_argument("--output_path", type=str)
    return parser.parse_args()

if __name__ == '__main__':
    if torch.cuda.is_available():
        print ('Cuda is available.')
    is_cuda = torch.cuda.is_available()

    args = parse_config()
    device = torch.device('cuda')
    model, tokenizer = load_model_tokenizer(args.model_name)
    if is_cuda:
        model = model.cuda(device)
    model.eval()

    text_list = []
    with open(args.file_path, 'r', encoding = 'utf8') as i:
        lines = i.readlines()
        for l in lines:
            text_list.append(l.strip('\n'))
            

    
    print ('Start measuring intra-sentence similarity...')
    print ('Number of text is {}'.format(len(text_list)))
    all_res_dict = {}

    import progressbar
    p = progressbar.ProgressBar(len(text_list))
    p.start()
    for idx in range(len(text_list)):
        p.update(idx)
        one_text = text_list[idx]
        one_res_dict = process_one_instance(model, tokenizer, one_text, is_cuda, device)
        all_res_dict[idx] = one_res_dict
    p.finish()
    print ('Finished!')

    import json
    with open(args.output_path, 'w') as outfile:
        json.dump(all_res_dict, outfile, indent=4)


