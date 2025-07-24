# -*- coding: utf-8 -*-
"""
Created on Tue Jun 15 18:04:08 2021

@author: wangl4
"""
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 15 16:28:14 2021

@author: wangl4
"""

# -*- coding: utf-8 -*-
"""
Created on Wed May 26 12:08:12 2021

@author: -
"""
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 19 05:22:22 2021

@author: wangl4
"""
import torch
import sys
from .base import BaseModel
from torch import nn
from torch import nn as nn
import torch

from utils import fix_random_seed_as

from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torch.nn import LayerNorm
FLOAT_MIN = -sys.float_info.max

class decoder(nn.Module):
    def __init__(self, hidden_units, n_items, token, dev):
        super(decoder, self).__init__()
        self.hidden_size = hidden_units
        self.num_items = n_items
        self.dev = dev
        self.token = token

    def forward(self, x):
        all_embs = self.token.weight[:self.num_items+1]
        logits = torch.matmul(x, torch.transpose(all_embs, 0, 1))
        #print(logits.shape)####B * T * item
        return logits

class SASRecWModel(BaseModel): # similar to torch.nn.MultiheadAttention
    def __init__(self, args):
        super().__init__(args)
        fix_random_seed_as(args.model_init_seed)

        self.n_layers = args.num_blocks
        self.heads = args.num_heads

        self.num_items = args.num_items
        self.dropout = args.dropout_rate
        self.dropout_emb = args.dropout_rate_emb
        self.item_emb_dropout = torch.nn.Dropout(p=self.dropout_emb)

        encoder_layers = TransformerEncoderLayer(self.hidden, self.heads, self.hidden*4, self.dropout, activation='gelu')
        encoder_norm = LayerNorm(self.hidden, eps=1e-8)
        self.transformer_encoder_t = TransformerEncoder(encoder_layers, self.n_layers, encoder_norm)
        self.decoder = decoder(self.hidden, self.num_items, self.token, self.dev)
        self.apply(self.init_weights)

    
    def seq2feats(self,x_t):
            padding_mask = (x_t == 0).squeeze(1)
            tl = x_t.shape[1]
            casual_mask = torch.triu(torch.ones(tl, tl) * float(-1e9), diagonal=1).to(self.dev)
            x_t = self.item_emb_dropout(self.token(x_t) + self.position_t(x_t))
            x_t = torch.transpose(x_t, 0, 1) # changed (N T E) to (T, N, E)
            x_t = self.transformer_encoder_t(x_t, mask=casual_mask, src_key_padding_mask=padding_mask)
            x_t = torch.transpose(x_t, 0, 1) # changed back to (N T E)
            return x_t    

    @classmethod
    def code(cls):
        return 'SASRecW'

    def forward(self, x_t): # for training
     
        log_feat_t = self.seq2feats(x_t)
        logit_t =self.decoder(log_feat_t)
        return log_feat_t,logit_t








