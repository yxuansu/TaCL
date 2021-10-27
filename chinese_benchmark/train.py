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

def save_model(model, save_path, save_name):
    from operator import itemgetter
    if not os.path.exists(save_path):
        os.mkdir(save_path)

    if torch.cuda.device_count() > 1: # multi-gpu training
        model = model.module

    model_save_path = save_path + '/' + save_name
    torch.save({'model':model.state_dict()}, model_save_path)

    fileData = {}
    for fname in os.listdir(save_path):
        if fname.startswith('epoch'):
            fileData[fname] = os.stat(save_path + '/' + fname).st_mtime
        else:
            pass
    sortedFiles = sorted(fileData.items(), key=itemgetter(1))
    if len(sortedFiles) < 1:
        pass
    else:
        delete = len(sortedFiles) - 1
        for x in range(0, delete):
            os.remove(save_path + '/' + sortedFiles[x][0])

def evaluate_model(args, data, model, save_path, mode):
    cuda_available = torch.cuda.is_available()
    if cuda_available: 
        if torch.cuda.device_count() > 1: # multi-gpu training 
            model = model.module
        else: # single gpu training
            pass
    else:
        pass

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

def train_one_model(args, model_name, run_number):
    save_path = args.save_path_prefix + '/run_{}'.format(run_number) + '/'
    import os
    if os.path.exists(save_path):
        pass
    else: # recursively construct directory
        os.makedirs(save_path, exist_ok=True)

    cuda_available = torch.cuda.is_available()
    device = torch.device('cuda')

    from transformers import BertTokenizer
    tokenizer = BertTokenizer.from_pretrained(model_name)

    print ('Loading data...')
    from dataclass import Data
    train_path, dev_path, test_path, label_path = args.train_path, args.dev_path, args.test_path, args.label_path
    data = Data(tokenizer, train_path, dev_path, test_path, label_path, args.max_len)
    print ('Data loaded.')

    print ('Loading model...')
    from model import NERModel
    model = NERModel(model_name, data.num_class)
    #if cuda_available:
    #    model = model.cuda(device)
    if cuda_available: 
        if torch.cuda.device_count() > 1: # multi-gpu training 
            print ('Multi-GPU training...')
            model = nn.DataParallel(model)
        else: # single gpu training
            pass
        model = model.to(device)
    else:
        pass

    print ('Model loaded.')
    model.train()

    optimizer = torch.optim.Adam(model.parameters(), args.learning_rate)
    optimizer.zero_grad()

    batch_size, gradient_accumulation_steps = args.batch_size, args.gradient_accumulation_steps
    train_num, dev_num, test_num = data.train_num, data.dev_num, data.test_num
    train_step_num, dev_step_num, test_step_num = int(train_num/batch_size) + 1, \
    int(dev_num/batch_size) + 1, int(test_num/batch_size) + 1

    print_every = int(train_step_num/4)

    batches_processed = 0
    loss_acm = 0.
    max_combine_score, best_combine_str = 0., 'best combine dev f1: {}, test f1: {}'.format(0., 0.)
    dev_f1_list, test_f1_list = [0.], [0.]
    best_combined_score_dict = {'dev':0., 'test':0.}
    max_test_f1_score = 0.
    for epoch_num in range(args.total_epochs):
        print ('------------------------------------------------------------------')
        print ('Start epoch {} training...'.format(epoch_num))
        model.train()
        p = progressbar.ProgressBar(train_step_num)
        p.start()
        for train_step in range(train_step_num):
            p.update(train_step)
            batches_processed += 1
            train_src_tensor, train_src_attn_mask, train_tgt_tensor, train_tgt_mask = data.get_next_train_batch(batch_size)
            if cuda_available:
                train_src_tensor = train_src_tensor.cuda(device)
                train_src_attn_mask = train_src_attn_mask.cuda(device)
                train_tgt_tensor = train_tgt_tensor.cuda(device)
                train_tgt_mask = train_tgt_mask.cuda(device)
            loss = model(train_src_tensor, train_src_attn_mask, train_tgt_tensor, train_tgt_mask)
            loss = loss.mean()
            loss_acm += loss.item()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

            if batches_processed % args.gradient_accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()

            if batches_processed % print_every == 0:
                one_loss = loss_acm / print_every
                one_loss = round(one_loss, 3)
                print ("epoch {}, batch {}, loss is {}".format(epoch_num, batches_processed, one_loss))
                print ("Batch %d, loss %.5f" % (batches_processed, loss_acm / batches_processed))
                loss_acm = 0.
        p.finish()

        model.eval()
        with torch.no_grad():
            _, _, dev_f1 = evaluate_model(args, data, model, save_path, mode='dev')
            _, _, test_f1 = evaluate_model(args, data, model, save_path, mode='test')
        model.train()

        dev_f1, test_f1 = dev_f1*100, test_f1*100
        dev_f1, test_f1 = round(dev_f1, 3), round(test_f1, 3)
        dev_f1_list.append(dev_f1)
        test_f1_list.append(test_f1)
        print ('At epoch {}, dev f1: {}, test f1: {}'.format(epoch_num, dev_f1, test_f1))
        one_combine_score = dev_f1 + test_f1
        if test_f1 > max_test_f1_score:
            best_combine_str = 'dev f1: {}, test f1: {}'.format(dev_f1, test_f1)
            best_combined_score_dict['dev'] = dev_f1
            best_combined_score_dict['test'] = test_f1
            max_dev_f1, max_test_f1 = max(dev_f1_list), max(test_f1_list)
            save_name = 'epoch_{}_dev_f1_{}_test_f1_{}_max_dev_f1_{}_max_test_f1_{}'.format(epoch_num, 
                dev_f1, test_f1, max_dev_f1, max_test_f1)
            save_model(model, save_path, save_name)
            max_combine_score = one_combine_score
            max_test_f1_score = test_f1

        print ('Current best combine result is ' + best_combine_str)
        print ('Best dev f1: {}, test f1: {}'.format(max(dev_f1_list), max(test_f1_list)))
        print ('Epoch {} finished.'.format(epoch_num))

    best_dev_f1, best_test_f1 = max(dev_f1_list), max(test_f1_list)
    best_combine_dev_f1, best_combine_test_f1 = best_combined_score_dict['dev'], best_combined_score_dict['test']
    return best_combine_dev_f1, best_combine_test_f1, best_dev_f1, best_test_f1

import numpy as np
def compute_mean_std(num_list):
    return round(np.mean(num_list), 2), round(np.std(num_list), 2)

def multiple_runs(args):
    model_path = args.model_name
    print ('------------------------------------------')
    print ('Evaluatiing model {}'.format(args.model_name))
    combine_dev_f1_list, combine_test_f1_list, best_dev_f1_list, best_test_f1_list = [], [], [], []
    for run in range(args.number_of_runs):
        print ('######')
        print ('start run {}'.format(run))
        one_best_combine_dev_f1, one_best_combine_test_f1, one_best_dev_f1, one_best_test_f1 = \
        train_one_model(args, model_path, run)
        combine_dev_f1_list.append(one_best_combine_dev_f1)
        combine_test_f1_list.append(one_best_combine_test_f1) 
        best_dev_f1_list.append(one_best_dev_f1) 
        best_test_f1_list.append(one_best_test_f1)

    combine_dev_f1_mean, combine_dev_f1_std = compute_mean_std(combine_dev_f1_list)
    combine_test_f1_mean, combine_test_f1_std = compute_mean_std(combine_test_f1_list)
    best_dev_f1_mean, best_dev_f1_std = compute_mean_std(best_dev_f1_list)
    best_test_f1_mean, best_test_f1_std = compute_mean_std(best_test_f1_list)
    overall_save_name = 'overall_combine_dev_f1_mean_{}_std_{}_test_f1_mean_{}_std_{}_best_dev_f1_mean_{}_std_{}_test_f1_mean_{}_std_{}'.format(
        combine_dev_f1_mean, combine_dev_f1_std, combine_test_f1_mean, combine_test_f1_std, best_dev_f1_mean, best_dev_f1_std, 
        best_test_f1_mean, best_test_f1_std)

def parse_config():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str)
    parser.add_argument("--train_path", type=str)
    parser.add_argument("--dev_path", type=str)
    parser.add_argument("--test_path", type=str)
    parser.add_argument("--label_path", type=str)
    parser.add_argument("--max_len", type=str, default=128)
    # learning configuration
    parser.add_argument("--number_of_runs", type=int, default=5, help="number of different experiment runs")
    parser.add_argument("--learning_rate", type=float)
    parser.add_argument("--batch_size_per_gpu", type=int)
    parser.add_argument("--number_of_gpu", type=int)
    parser.add_argument("--batch_size", type=int)
    parser.add_argument("--gradient_accumulation_steps", type=int, help="gradient accumulation step.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument("--total_epochs", type=int)
    parser.add_argument("--save_path_prefix", type=str, help="directory to save the model evaluation results.")
    parser.add_argument("--evaluation_mode", type=str, default="BMES", help="BMES or BIO")
    return parser.parse_args()

import argparse
if __name__ == '__main__':
    if torch.cuda.is_available():
        print ('Cuda is available.')
    cuda_available = torch.cuda.is_available()
    device = torch.device('cuda')

    args = parse_config()

    assert args.batch_size_per_gpu * args.number_of_gpu == args.batch_size

    multiple_runs(args)

        