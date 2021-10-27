import os
import sys
import torch
import random
from torch import nn
import torch.nn.functional as F
from metric_py3 import fmeasure_from_singlefile
import progressbar

def combine_result(gold_path, pred_path, out_path):
    with open(out_path, 'w', encoding = 'utf8') as o:
        with open(gold_path, 'r', encoding = 'utf8') as g:
            gold_lines = g.readlines()
        with open(pred_path, 'r', encoding = 'utf8') as p:
            pred_lines = p.readlines()
        assert len(gold_lines) == len(pred_lines)
        data_num = len(gold_lines)
        for i in range(data_num):
            gold_l = gold_lines[i]
            pred_l = pred_lines[i]
            gold_content_list = gold_l.strip('\n').split('\t')
            text = gold_content_list[0]
            gold_label_str = gold_content_list[1]
            
            pred_l = pred_lines[i]
            pred_content_list = pred_l.strip('\n').split('\t')
            pred_label_str = pred_content_list[1]
            
            pred_label_list = pred_label_str.split()
            gold_label_list = gold_label_str.split()[:len(pred_label_list)] # result truncation
            assert len(gold_label_list) == len(pred_label_list)
            
            instance_len = len(gold_label_list)
            text_list = text.split()[:instance_len]
            for j in range(instance_len):
                out_str = text_list[j] + ' ' + gold_label_list[j] + ' ' + pred_label_list[j]
                o.writelines(out_str + '\n')
            o.writelines('\n')

def evaluate_model(args, data, model, save_path, mode):
    import os
    if os.path.exists(save_path):
        pass
    else: # recursively construct directory
        os.makedirs(save_path, exist_ok=True)

    cuda_available = torch.cuda.is_available()


    device = torch.device('cuda')
    if mode == 'dev':
        eval_step_num = int(data.dev_num/args.batch_size) + 1
        instance_num = data.dev_num
        gold_path = data.dev_path
    elif mode == 'test':
        eval_step_num = int(data.test_num/args.batch_size) + 1
        instance_num = data.test_num
        gold_path = data.test_path
    else:
        raise Exception('Wrong Mode!!!')

    res_list = []
    with torch.no_grad():
        model.eval()
        for _ in range(eval_step_num):
            src_tensor, src_attn_mask, _, tgt_mask, tgt_ref_id_list = \
            data.get_next_validation_batch(args.batch_size, mode)
            if cuda_available:
                src_tensor = src_tensor.cuda(device)
                src_attn_mask = src_attn_mask.cuda(device)
                tgt_mask = tgt_mask.cuda(device)
            predictions = model.decode(src_tensor, src_attn_mask, tgt_mask)
            predictions = data.parse_result(predictions)
            ref_predictions = data.parse_result(tgt_ref_id_list)
            bsz = len(tgt_ref_id_list)
            for idx in range(bsz):
                assert len(predictions[idx].split()) == len(ref_predictions[idx].split())
            res_list += predictions
        res_list = res_list[:instance_num]
    eval_path = save_path + '/eval.txt'
    with open(eval_path, 'w', encoding = 'utf8') as o:
        for res in res_list:
            o.writelines(res + '\t' + res + '\n')
    combine_path = save_path + '/' + mode + '_gold_eval_combine.txt'
    combine_result(gold_path, eval_path, combine_path)
    precision, recall, f1 = fmeasure_from_singlefile(combine_path, args.evaluation_mode)
    os.remove(combine_path)
    os.remove(eval_path)
    return precision, recall, f1


def parse_config():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="cambridgeltl/clbert-base-chinese")
    parser.add_argument("--saved_ckpt_path", type=str, help="the path of the trained model.")
    parser.add_argument("--train_path", type=str)
    parser.add_argument("--dev_path", type=str)
    parser.add_argument("--test_path", type=str)
    parser.add_argument("--label_path", type=str)
    parser.add_argument("--max_len", type=str, default=128)
    # learning configuration
    parser.add_argument("--batch_size", type=int)
    parser.add_argument("--evaluation_mode", type=str, default="BMES", help="BMES or BIO")
    return parser.parse_args()

import argparse
if __name__ == '__main__':
    if torch.cuda.is_available():
        print ('Cuda is available.')
    cuda_available = torch.cuda.is_available()
    device = torch.device('cuda')

    args = parse_config()

    from transformers import BertTokenizer
    tokenizer = BertTokenizer.from_pretrained(args.model_name)

    print ('Loading data...')
    from dataclass import Data
    train_path, dev_path, test_path, label_path = args.train_path, args.dev_path, args.test_path, args.label_path
    data = Data(tokenizer, train_path, dev_path, test_path, label_path, args.max_len)
    print ('Data loaded.')

    print ('Loading model...')
    from model import NERModel
    model = NERModel(args.model_name, data.num_class)
    model_ckpt = torch.load(args.saved_ckpt_path)
    model_parameters = model_ckpt['model']
    model.load_state_dict(model_parameters)
    model = model.to(device)
    print ('Model Loaded.')

    with torch.no_grad():
        save_path = r'./eval_folder/'
        test_precision, test_recall, test_f1 = evaluate_model(args, data, model, save_path, mode='test')

    test_precision, test_recall, test_f1 = round(test_precision, 3), round(test_recall, 3), round(test_f1, 3)
    print ('------------------------------------------------------------')
    print ('Test Evaluation Results are Precision: {}, Recall: {}, F1: {}'.format(test_precision, test_recall, test_f1))
