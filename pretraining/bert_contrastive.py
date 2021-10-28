import sys
import os
import operator
from operator import itemgetter
import torch
from torch import nn
import torch.nn.functional as F
import random
import numpy as np
import argparse

from transformers import BertForPreTraining, BertTokenizer, BertConfig, BertModel
from torch.nn import CrossEntropyLoss

def label_smoothed_nll_loss(contrastive_scores, contrastive_labels, eps=0.0):
    '''
        contrasive_scores: bsz x seqlen x seqlen
        contrasive_labels: bsz x seqlen; masked positions with 0., otherwise 1.
    '''
    bsz, seqlen, _ = contrastive_scores.size()
    logprobs = F.log_softmax(contrastive_scores.view(-1, seqlen), dim=-1)
    gold = torch.arange(seqlen).view(-1,)
    gold = gold.expand(bsz, seqlen).contiguous().view(-1)
    if contrastive_scores.is_cuda:
        gold = gold.cuda(contrastive_scores.get_device())
    loss =  -logprobs.gather(dim=-1, index=gold.unsqueeze(1)).squeeze(1)
    loss = loss.view(bsz, seqlen) * contrastive_labels
    loss = torch.sum(loss) / contrastive_labels.sum()

    _, pred = torch.max(logprobs, -1)
    correct_num = torch.eq(gold, pred).float().view(bsz, seqlen)
    correct_num = torch.sum(correct_num * contrastive_labels)
    total_num = contrastive_labels.sum()
    return loss, correct_num, total_num

class BERTContrastivePretraining(nn.Module):
    def __init__(self, model_name, sim='cosine', temperature=0.01, use_contrastive_loss='True'):
        super(BERTContrastivePretraining, self).__init__()
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.vocab_size = self.tokenizer.vocab_size
        self.model = BertForPreTraining.from_pretrained(model_name)
        self.bert = self.model.bert
        self.cls = self.model.cls
        self.config = BertConfig.from_pretrained(model_name)
        embed_dim = self.config.hidden_size
        self.embed_dim = embed_dim
        self.loss_fct = CrossEntropyLoss(ignore_index=-100)
        assert sim in ['dot_product', 'cosine']
        self.sim = sim
        if self.sim == 'dot_product':
            print ('use dot product similarity')
        else:
            print ('use cosine similarity')
        self.temperature = temperature

        if use_contrastive_loss == 'True':
            use_contrastive_loss = True
        elif use_contrastive_loss == 'False':
            use_contrastive_loss = False
        else:
            raise Exception('Wrong contrastive loss setting!')

        self.use_contrastive_loss = use_contrastive_loss
        if self.use_contrastive_loss:
            print ('Initializing teacher BERT.')
            self.teacher_bert = BertModel.from_pretrained(model_name)
            for param in self.teacher_bert.parameters():
                param.requires_grad = False
            print ('Teacher BERT initialized.')
        else:
            print ('Train BERT with vanilla MLM loss.')

    def save_model(self, ckpt_save_path):
        import os
        if os.path.exists(ckpt_save_path):
            pass
        else: # recursively construct directory
            os.makedirs(ckpt_save_path, exist_ok=True)

        # save model
        self.bert.save_pretrained(ckpt_save_path)
        # save tokenizer
        self.tokenizer.save_pretrained(ckpt_save_path)

    def compute_teacher_representations(self, input_ids, token_type_ids, attention_mask):
        bsz, seqlen = input_ids.size()
        outputs = self.teacher_bert(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)
        rep, pooled_output = outputs[0], outputs[1]
        # rep: bsz x seqlen x embed_dim
        rep = rep.view(bsz, seqlen, self.embed_dim)
        logits, sen_relation_scores = self.cls(rep, pooled_output) # bsz x seqlen x vocab_size
        return rep, logits, sen_relation_scores

    def compute_representations(self, input_ids, token_type_ids, attention_mask):
        bsz, seqlen = input_ids.size()
        outputs = self.bert(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)
        rep, pooled_output = outputs[0], outputs[1]
        # rep: bsz x seqlen x embed_dim
        rep = rep.view(bsz, seqlen, self.embed_dim)
        logits, sen_relation_scores = self.cls(rep, pooled_output) # bsz x seqlen x vocab_size
        return rep, logits, sen_relation_scores

    def compute_mlm_loss(self, truth, msk, logits):
        truth = truth.transpose(0,1)
        msk = msk.transpose(0,1)
        msk_token_num = torch.sum(msk).float().item()
        # center
        y_mlm = logits.transpose(0,1).masked_select(msk.unsqueeze(-1).to(torch.bool))
        y_mlm = y_mlm.view(-1, self.vocab_size)
        gold = truth.masked_select(msk.to(torch.bool))
        log_probs_mlm = torch.log_softmax(y_mlm, -1)
        mlm_loss = F.nll_loss(log_probs_mlm, gold, reduction='mean')
        _, pred_mlm = log_probs_mlm.max(-1)
        mlm_correct_num = torch.eq(pred_mlm, gold).float().sum().item()
        return mlm_loss, mlm_correct_num

    def forward(self, truth, inp, seg, msk, attn_msk, labels, contrastive_labels, nxt_snt_flag):
        '''
           truth: bsz x seqlen
           inp: bsz x seqlen
           seg: bsz x seqlen
           msk: bsz x seqlen
           attn_msk: bsz x seqlen
           labels: bsz x seqlen; masked positions are filled with -100
           contrastive_labels: bsz x seqlen; masked position with 0., otherwise 1.
        '''
        if truth.is_cuda:
            is_cuda = True
            device = truth.get_device()
        else:
            is_cuda = False

        bsz, seqlen = truth.size()
        masked_rep, logits, sen_relation_scores = \
        self.compute_representations(input_ids=inp, token_type_ids=seg, attention_mask=attn_msk)

        # compute masked language model loss
        # --------------------------------------------------------------------------------------- #
        mlm_loss, mlm_correct_num = self.compute_mlm_loss(truth, msk, logits)
        # --------------------------------------------------------------------------------------- #

        # compute contrastive loss
        if self.use_contrastive_loss:
            truth_rep, truth_logits, _ =  self.compute_teacher_representations(input_ids=truth, token_type_ids=seg, attention_mask=attn_msk)
            ''' 
                mask_rep, truth_rep : hidden_size
                rep, left_rep, right_rep: bsz x seqlen x embed_dim
            '''
            if self.sim == 'dot_product':
                contrastive_scores = torch.matmul(masked_rep, truth_rep.transpose(1,2))
                
            elif self.sim == 'cosine': # 'cosine'
                masked_rep = masked_rep / masked_rep.norm(dim=2, keepdim=True)
                truth_rep = truth_rep / truth_rep.norm(dim=2, keepdim=True)
                contrastive_scores = torch.matmul(masked_rep, truth_rep.transpose(1,2)) / self.temperature # bsz x seqlen x seqlen
            else:
                raise Exception('Wrong similarity mode!!!')

            assert contrastive_scores.size() == torch.Size([bsz, seqlen, seqlen])
            contrastive_loss, correct_contrastive_num, total_contrastive_num = \
            label_smoothed_nll_loss(contrastive_scores, contrastive_labels)
        else:
            correct_contrastive_num, total_contrastive_num = 0., 1.

        if is_cuda:
            nxt_snt_flag = nxt_snt_flag.type(torch.LongTensor).cuda(device)
        else:
            nxt_snt_flag = nxt_snt_flag.type(torch.LongTensor)
        next_sentence_loss = self.loss_fct(sen_relation_scores.view(-1, 2), nxt_snt_flag.view(-1))
        if self.use_contrastive_loss:
            tot_loss = mlm_loss + next_sentence_loss + contrastive_loss
        else:
            tot_loss = mlm_loss + next_sentence_loss

        next_sentence_logprob = torch.log_softmax(sen_relation_scores, -1)
        next_sentence_predictions = torch.max(next_sentence_logprob, dim = -1)[-1]
        nxt_snt_correct_num = torch.eq(next_sentence_predictions, nxt_snt_flag.view(-1)).float().sum().item()
        tot_tokens = msk.float().sum().item()
        if is_cuda:
            return tot_loss, torch.Tensor([mlm_correct_num]).cuda(device), torch.Tensor([tot_tokens]).cuda(device), \
            torch.Tensor([nxt_snt_correct_num]).cuda(device), torch.Tensor([correct_contrastive_num]).cuda(device),\
            torch.Tensor([total_contrastive_num]).cuda(device)
        else:
            return tot_loss, torch.Tensor([mlm_correct_num]), torch.Tensor([tot_tokens]), torch.Tensor([nxt_snt_correct_num]), \
            torch.Tensor([correct_contrastive_num]), torch.Tensor([total_contrastive_num])

        
