import os
import sys
import torch
import random
import argparse
import operator
import numpy as np
from torch import nn
from TorchCRF import CRF
import torch.nn.functional as F
from operator import itemgetter
from transformers import BertModel, BertTokenizer, BertConfig

class NERModel(nn.Module):
    def __init__(self, model_name, num_class):
        super(NERModel, self).__init__()
        self.num_class = num_class
        self.crf = CRF(self.num_class)
        self.bert = BertModel.from_pretrained(model_name)
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.config = BertConfig(model_name)
        self.embed_dim = self.config.hidden_size
        self.fc = nn.Linear(self.embed_dim, self.num_class)

    def compute_logits(self, src_tensor, src_attn_mask):
        representation = self.bert(input_ids=src_tensor, attention_mask=src_attn_mask)[0] # bsz x seqlen x hidden_size
        bsz, seqlen, _ = representation.size()
        logits = self.fc(representation.view(bsz * seqlen, self.embed_dim)).view(bsz, seqlen, self.num_class)
        return logits

    def forward(self, src_tensor, src_attn_mask, tgt_tensor, tgt_mask):
        logits = self.compute_logits(src_tensor, src_attn_mask)
        loglikelihood = self.crf.forward(logits, tgt_tensor, tgt_mask).mean()
        loss = -1 * loglikelihood
        return loss

    def decode(self, src_tensor, src_attn_mask, tgt_mask):
        logits = self.compute_logits(src_tensor, src_attn_mask)
        res = self.crf.viterbi_decode(logits, tgt_mask)
        return res

