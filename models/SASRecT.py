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
from models.base import BaseModel
from torch import nn
from torch import nn as nn
import torch

from utils import fix_random_seed_as

from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torch.nn import LayerNorm
from torch.nn import ReLU

FLOAT_MIN = -sys.float_info.max
class PositionwiseFeedForward(nn.Module):
    '''Implements FFN equation.'''

    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = ReLU()
        self.laynorm = LayerNorm(d_ff)

    def forward(self, x):
        return self.w_2(self.dropout(self.activation(self.laynorm(self.w_1(x)))))

class decoder(nn.Module):
    def __init__(self, hidden_units, n_items, token, dev):
        super(decoder, self).__init__()
        self.hidden_size = hidden_units
        self.num_items = n_items
        self.dev = dev
        self.token = token
        self.linear1 = nn.Linear(self.hidden_size, self.hidden_size)
        self.activation = nn.GELU()
        self.LayerNorm = nn.LayerNorm(self.hidden_size)

    def forward(self, x):
    
        all_embs = self.token.weight[:self.num_items+1]
        
        logits = torch.matmul(x, torch.transpose(all_embs, 0, 1))
        # x = self.LayerNorm(self.activation(self.linear1(x)))
        # bias = nn.Parameter(torch.zeros(self.num_items+1).to(self.dev))
        # logits = torch.matmul(x, torch.transpose(all_embs, 0, 1)) + bias
        return logits

# class SASRecTSModel(nn.Module): # 
#     def __init__(self, args, token, position,hidden,dropout_rate,dropout_rate_emb, device):
#         super(SASRecTSModel,self).__init__()

#         self.n_layers = args.num_blocks
#         self.heads = args.num_heads
#         self.token = token
#         self.position = position
#         self.hidden = hidden
#         self.dev  = device
#         self.num_items = args.num_items
#         self.dev = args.device
#         self.dropout = dropout_rate
#         self.dropout_emb = dropout_rate_emb
#         self.item_emb_dropout = torch.nn.Dropout(p=self.dropout_emb)

#         encoder_layers = TransformerEncoderLayer(self.hidden, self.heads, self.hidden*4, self.dropout, activation='gelu')
#         encoder_norm = LayerNorm(self.hidden, eps=1e-8)
#         self.transformer_encoder_s = TransformerEncoder(encoder_layers, self.n_layers, encoder_norm)
#         self.decoder = decoder(self.hidden,self.num_items,self.token,self.dev)

    
#     def seq2feats(self, x_s):
#             padding_mask = (x_s == 0).squeeze(1)
#             tl = x_s.shape[1]
#             casual_mask = torch.triu(torch.ones(tl, tl) * float(-1e9), diagonal=1).to(self.dev)
#             x_s = self.item_emb_dropout(self.token(x_s) + self.position(x_s))
#             x_s = torch.transpose(x_s, 0, 1) # changed (N T E) to (T, N, E)
#             x_s = self.transformer_encoder_s(x_s, mask=casual_mask, src_key_padding_mask=padding_mask)
#             x_s = torch.transpose(x_s, 0, 1) # changed back to (N T E)
#             return x_s  

#     def forward(self, x_s): # for training

#         log_feat_t = self.seq2feats(x_s)
#         logit_t =self.decoder(log_feat_t)

#         return log_feat_t,logit_t

class SASRecTTModel(nn.Module): # similar to torch.nn.MultiheadAttention
    def __init__(self, args, token, position,hidden, dropout_rate,dropout_rate_emb,device):
        super(SASRecTTModel,self).__init__()

        self.n_layers = args.num_blocks
        self.heads = args.num_heads
        self.token = token
        self.position = position
        
        self.hidden = hidden
        self.dev = device
        self.num_items = args.num_items
        self.dropout = dropout_rate
        self.dropout_emb = dropout_rate_emb 
        self.item_emb_dropout = torch.nn.Dropout(p=self.dropout_emb)

        encoder_layers = TransformerEncoderLayer(self.hidden, self.heads, self.hidden*4, self.dropout, activation='gelu')
        encoder_norm = LayerNorm(self.hidden, eps=1e-8)
        self.transformer_encoder_s = TransformerEncoder(encoder_layers, self.n_layers, encoder_norm)
        self.decoder = decoder(self.hidden,self.num_items,self.token,self.dev)
    
    def seq2feats(self, x_s):
            padding_mask = (x_s == 0).squeeze(1)
            tl = x_s.shape[1]
            casual_mask = torch.triu(torch.ones(tl, tl) * float(-1e9), diagonal=1).to(self.dev)
            x_s = self.item_emb_dropout(self.token(x_s) + self.position(x_s))
            x_s = torch.transpose(x_s, 0, 1) # changed (N T E) to (T, N, E)
            x_s = self.transformer_encoder_s(x_s, mask=casual_mask, src_key_padding_mask=padding_mask)
            x_s = torch.transpose(x_s, 0, 1) # changed back to (N T E)
            return x_s  

    def forward(self, x_s): # for training

        log_feat_t = self.seq2feats(x_s)
        logit_t =self.decoder(log_feat_t)

        return log_feat_t,logit_t

class Model(BaseModel): # similar to torch.nn.MultiheadAttention
    def __init__(self, args):
        super().__init__(args)
        fix_random_seed_as(args.model_init_seed)
        self.dropout_rate=args.dropout_rate
        self.dropout_rate_emb_s = args.dropout_rate_emb
        self.dropout_rate_emb_t = args.dropout_rate_emb_t
        self.model_s = SASRecTTModel(args,self.token,self.position_t,self.hidden,self.dropout_rate,self.dropout_rate_emb_s,self.dev)
        self.model_t = SASRecTTModel(args,self.token,self.position_t,self.hidden,self.dropout_rate,self.dropout_rate_emb_t,self.dev)
        self.FFN_1 = PositionwiseFeedForward(self.hidden, self.hidden, self.dropout_rate)
        #self.FFN_2 = PositionwiseFeedForward(self.hidden, self.hidden, self.dropout_rate)
        self.apply(self.init_weights)  

    @classmethod
    def code(cls):
        return 'SASRecT'

    def forward(self, x_s,x_s_s,x_t): # for training

        log_feat_s,logit_s = self.model_s(x_s)
        log_feat_s_1,logit_s_1 = self.model_s(x_s)
        #log_feat_s = self.FFN_1(log_feat_s)
        log_feat_s_s,logit_s_s = self.model_t(x_s_s)
        #log_feat_s_s = self.FFN_1(log_feat_s_s)
        #print(x_s_s.size())
        log_feat_t,logit_t = self.model_t(x_t)
        #log_feat_t = self.FFN_1(log_feat_t)

        return log_feat_s,logit_s, log_feat_s_1,logit_s_1,log_feat_s_s, logit_s_s, log_feat_t,logit_t






