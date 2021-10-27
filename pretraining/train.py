# coding=utf-8
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import torch.multiprocessing as mp
import argparse, os
import random

import time
import logging

def parse_config():
    parser = argparse.ArgumentParser()
    parser.add_argument("--language", type=str, help="the language of your pre-trained model. chinese or english")
    parser.add_argument("--model_name", type=str, 
        help="the pre-trained model specification. bert-base-uncased or bert-base-chinese")
    parser.add_argument("--sim", type=str, default="cosine", 
        help='similarity measurement for contrastive masked language model; dot_product or cosine')
    parser.add_argument("--temperature", type=float, default=0.01, help='temperature setting when using cosine similarity')
    parser.add_argument("--use_contrastive_loss", type=str, default='True', help='whether use contrastive loss in training.')

    # data configuration
    parser.add_argument("--whole_word_masking", type=str, default='False', 
        help="whether apply whole word masking during pretraining.")
    parser.add_argument("--train_data", type=str, help="path to the pre-training data")
    parser.add_argument("--max_len", type=int, help="maximum length for each sequence.")

    # mini-batch training configuration
    parser.add_argument("--number_of_gpu", type=int, help="Number of available GPUs.")  
    parser.add_argument("--batch_size_per_gpu", type=int, help='batch size for each gpu.') 
    parser.add_argument("--gradient_accumulation_steps", type=int, help="gradient accumulation step.")
    parser.add_argument("--effective_batch_size", type=int, 
        help="effective_bsz = batch_size_per_gpu x number_of_gpu x gradient_accumulation_steps")
    parser.add_argument("--total_steps", type=int, help="total effective training steps")
    # learning configuration
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument('--print_every', type=int)
    parser.add_argument('--save_every', type=int)
    parser.add_argument("--ckpt_save_path", type=str, help="directory to save the model parameters.")
    return parser.parse_args()

def update_lr(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def train_model(args, model, checkpoint_save_prefix_path, gradient_accumulation_steps, tokenizer, 
        dataset_batch_size, max_len, whole_word_masking, total_steps):
    cuda_available = torch.cuda.is_available()
    multi_gpu_training = False
    if cuda_available:
        if torch.cuda.device_count() > 1:
            multi_gpu_training = True
        else:
            pass
    else:
        pass
    device = torch.device('cuda')

    warmup_steps = int(0.1 * total_steps) # 10% of training steps are used for warmup
    print ('total training steps is {}, warmup steps is {}'.format(total_steps, warmup_steps))

    from transformers.optimization import AdamW, get_linear_schedule_with_warmup
    optimizer = AdamW(model.parameters(), lr=args.learning_rate)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)
    optimizer.zero_grad()

    contrastive_acc_acm, ntokens_acm, acc_nxt_acm, npairs_acm = 0., 0., 0., 0.
    mlm_acm = 0.0
    train_loss = 0.

    effective_batch_acm = 0
    all_batch_step = 1
    print_valid, save_valid = False, False
    model.train()

    if args.language == 'chinese':
        from dataclass_chinese import PretrainCorpus
        print ('Train Chinese Model.')
    elif args.language == 'english':
        from dataclass_english import PretrainCorpus
        print ('Train English Model.')
    else:
        raise Exception("Wrong language specification")
    print ('Loading data...')
    data = PretrainCorpus(tokenizer, args.train_data, max_len, whole_word_masking=whole_word_masking)
    print ('Data loaded.')
    #for truth, inp, seg, msk, attn_mask, nxt_snt_flag in data:
    while effective_batch_acm < total_steps:
        #print (all_batch_step, gradient_accumulation_steps)
        truth, inp, seg, msk, attn_mask, labels, contrastive_labels, nxt_snt_flag = \
        data.get_batch_data(dataset_batch_size)
        all_batch_step += 1
        # have one small batch of data
        if effective_batch_acm <= warmup_steps:
            update_lr(optimizer, args.learning_rate*effective_batch_acm/warmup_steps)

        if cuda_available:
            truth = truth.cuda(device)
            inp = inp.cuda(device)
            seg = seg.cuda(device)
            msk = msk.cuda(device)
            attn_mask = attn_mask.cuda(device)
            nxt_snt_flag = nxt_snt_flag.cuda(device)
            labels = labels.cuda(device)
            contrastive_labels = contrastive_labels.cuda(device)
        bsz = truth.size()[0]

        loss, mlm_correct_num, tot_tokens, nxt_snt_correct_num, correct_contrastive_num, total_contrastive_num = \
        model(truth, inp, seg, msk, attn_mask, labels, contrastive_labels, nxt_snt_flag)

        mlm_correct_num = torch.sum(mlm_correct_num).item()
        tot_tokens = torch.sum(tot_tokens).item()
        nxt_snt_correct_num = torch.sum(nxt_snt_correct_num).item()
        correct_contrastive_num = torch.sum(correct_contrastive_num).item()

        # keep track of intermediate result
        ntokens_acm += tot_tokens
        mlm_acm += mlm_correct_num
        contrastive_acc_acm += correct_contrastive_num
        acc_nxt_acm += nxt_snt_correct_num
        npairs_acm += bsz
            
        loss = loss.mean()
        loss.backward()
        train_loss += loss.item()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

        # parameter update
        if all_batch_step % gradient_accumulation_steps == 0:
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            effective_batch_acm += 1
            print_valid, save_valid = True, True

        # print intermediate result
        if effective_batch_acm % args.print_every == 0 and print_valid:
            one_train_loss = train_loss / (effective_batch_acm * gradient_accumulation_steps)
            middle_acc, contrastive_acc = mlm_acm / ntokens_acm, contrastive_acc_acm / ntokens_acm
            nxt_snt_acc = acc_nxt_acm / npairs_acm

            middle_acc = round(middle_acc*100,3)
            contrastive_acc = round(contrastive_acc*100,3)
            nxt_snt_acc = round(nxt_snt_acc*100,3)

            print ('At training steps {}, training loss is {}, middle mlm acc is {}, contrastive acc is {}, \
                next sentence acc is {}'.format(effective_batch_acm, one_train_loss, middle_acc, contrastive_acc, nxt_snt_acc))
            print_valid = False

        # saving result
        if effective_batch_acm % args.save_every == 0 and save_valid:
            one_train_loss = train_loss / (args.save_every * gradient_accumulation_steps)
            middle_acc, contrastive_acc = mlm_acm / ntokens_acm, contrastive_acc_acm / ntokens_acm
            nxt_snt_acc = acc_nxt_acm / npairs_acm

            middle_acc = round(middle_acc*100,3)
            contrastive_acc = round(contrastive_acc*100,3)
            nxt_snt_acc = round(nxt_snt_acc*100,3)

            print ('At training steps {}, training loss is {}, middle mlm acc is {}, contrastive acc is {}, \
                next sentence acc is {}'.format(effective_batch_acm, one_train_loss, middle_acc, contrastive_acc, nxt_snt_acc))
            print ('Saving Model...')
            save_name = 'training_step_{}_middle_mlm_acc_{}_contrastive_acc_{}_nxt_sen_acc_{}'.format(effective_batch_acm,
                middle_acc, contrastive_acc, nxt_snt_acc)
            save_valid = False


            model_save_path = checkpoint_save_prefix_path + '/' + save_name
            import os
            if os.path.exists(model_save_path):
                pass
            else: # recursively construct directory
                os.makedirs(model_save_path, exist_ok=True)

            if cuda_available and torch.cuda.device_count() > 1:
                model.module.save_model(model_save_path)
            else:
                model.save_model(model_save_path)
            print ('Model Saved!')
            train_loss = 0.
            contrastive_acc_acm, ntokens_acm, acc_nxt_acm, npairs_acm = 0., 0., 0., 0.
            mlm_acm = 0.0
    print ('Training Finished!')
    print ('all_batch_step {}'.format(all_batch_step))
    return model

import argparse
if __name__ == '__main__':
    if torch.cuda.is_available():
        print ('Cuda is available.')
    cuda_available = torch.cuda.is_available()
    multi_gpu_training = False
    if cuda_available:
        if torch.cuda.device_count() > 1:
            multi_gpu_training = True
            print ('Using Multi-GPU training, number of GPU is {}'.format(torch.cuda.device_count()))
        else:
            print ('Using single GPU training.')
    else:
        pass

    args = parse_config()
    device = torch.device('cuda')
    model_name = args.model_name

    from bert_contrastive import BERTContrastivePretraining
    print ('Initializing Bert Model...')
    model = BERTContrastivePretraining(model_name, args.sim, args.temperature, args.use_contrastive_loss)

    if cuda_available:
        if multi_gpu_training:
            model = nn.DataParallel(model) # multi-gpu training
        else:
            pass
        model = model.to(device)
    else:
        pass
    print ('Bert model loaded') 

    from transformers import BertTokenizer
    if args.whole_word_masking == 'True':
        print ('Use whole word masking schema.')
        whole_word_masking = True
    elif args.whole_word_masking == 'False':
        print ('Use original masking schema.')
        whole_word_masking = False
    else:
        raise Exception('Wrong whole_word_masking configuration!!!')

    print ('--------------------------------------------------------------------------')
    print ('Start Training:')
    batch_size_per_gpu, gradient_accumulation_steps, number_of_gpu, effective_batch_size = \
    args.batch_size_per_gpu, args.gradient_accumulation_steps, args.number_of_gpu, args.effective_batch_size
    assert effective_batch_size == batch_size_per_gpu * gradient_accumulation_steps * number_of_gpu
    max_len = args.max_len

    print ('Effective batch size {}, maximum length {}'.format(effective_batch_size, max_len))
    tokenizer = BertTokenizer.from_pretrained(model_name)
    checkpoint_save_prefix_path = args.ckpt_save_path + '/'
    model = train_model(args, model, checkpoint_save_prefix_path, gradient_accumulation_steps, tokenizer, 
        batch_size_per_gpu * number_of_gpu, max_len, whole_word_masking, args.total_steps)
    print ('Training Finished!')

