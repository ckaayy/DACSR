import torch
import sys
from models.base import BaseModel
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
    
        # all_seqs = np.array(range(self.num_items+1))
        # all_seqs  = torch.LongTensor(all_seqs).to(self.dev)
        all_embs = self.token.weight[:self.num_items+1]
        #all_embs = self.token(all_seqs)#### item *hidden
        #print(all_embs.shape)

        logits = torch.matmul(x, torch.transpose(all_embs, 0, 1))
        #print(logits.shape)####B * T * item
        return logits


class SASRecTTModel(nn.Module): # similar to torch.nn.MultiheadAttention
    def __init__(self, args, token, position,hidden, dropout_rate,device):
        super(SASRecTTModel,self).__init__()

        self.n_layers = args.num_blocks
        self.heads = args.num_heads
        self.token = token
        self.position = position
        self.hidden = hidden
        self.dev = device
        self.num_items = args.num_items
        self.dropout = dropout_rate
        #self.dropout_emb = dropout_rate_emb 
        #self.item_emb_dropout = torch.nn.Dropout(p=self.dropout_emb)

        encoder_layers = TransformerEncoderLayer(self.hidden, self.heads, self.hidden*4, self.dropout, activation='gelu')
        encoder_norm = LayerNorm(self.hidden, eps=1e-8)
        self.transformer_encoder_s = TransformerEncoder(encoder_layers, self.n_layers, encoder_norm)
        self.decoder = decoder(self.hidden,self.num_items,self.token,self.dev)

    def forward(self, x_s, casual_mask,padding_mask): # for training
        x_s = torch.transpose(x_s, 0, 1) # changed (N T E) to (T, N, E)
        x_s = self.transformer_encoder_s(x_s, mask=casual_mask, src_key_padding_mask=padding_mask)
        x_s = torch.transpose(x_s, 0, 1)
        logit_t =self.decoder(x_s)

        return x_s,logit_t

class Model_1(BaseModel): # similar to torch.nn.MultiheadAttention
    def __init__(self, args):
        super().__init__(args)
        fix_random_seed_as(args.model_init_seed)
        self.model = SASRecTTModel(args,self.token,self.position_t,self.hidden,args.dropout_rate,self.dev)
        #self.model_t = SASRecTTModel(args,self.token_t,self.position,self.hidden,0.1,0.5, self.dev)
        self.dropout_emb_1 = args.dropout_rate_emb 
        self.item_emb_dropout_1 = torch.nn.Dropout(p=self.dropout_emb_1)
        self.dropout_emb_2 = args.dropout_rate_emb + 0.4
        self.item_emb_dropout_2 = torch.nn.Dropout(p=self.dropout_emb_2)
        self.apply(self.init_weights)  

    @classmethod
    def code(cls):
        return 'SASRecT_1'

    def seq2feats(self, x_s):
            padding_mask = (x_s == 0).squeeze(1)
            tl = x_s.shape[1]
            casual_mask = torch.triu(torch.ones(tl, tl) * float(-1e9), diagonal=1).to(self.dev)
            #x_s = self.item_emb_dropout(self.token(x_s) + self.position(x_s))
            return casual_mask, padding_mask

    def forward(self, x_s,x_s_s,x_t): # for trainin
        casual_mask_s, padding_mask_s = self.seq2feats(x_s)
        x_s = self.item_emb_dropout_1(self.token(x_s) + self.position_t(x_s))
        log_feat_s,logit_s = self.model(x_s, casual_mask_s, padding_mask_s)
        casual_mask_s_s, padding_mask_s_s = self.seq2feats(x_s_s)
        x_s_s = self.item_emb_dropout_1(self.token(x_s_s) + self.position_t(x_s_s))
        log_feat_s_s,logit_s_s = self.model(x_s_s, casual_mask_s_s, padding_mask_s_s)
        casual_mask_t, padding_mask_t = self.seq2feats(x_t)
        x_t = self.item_emb_dropout_1(self.token(x_t) + self.position_t(x_t))
        log_feat_t,logit_t = self.model(x_t, casual_mask_t, padding_mask_t)

        return log_feat_s,logit_s, log_feat_s_s, logit_s_s, log_feat_t,logit_t






