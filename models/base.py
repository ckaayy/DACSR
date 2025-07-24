import torch.nn as nn
import torch
import numpy as np
from models.emb_modules.embedding.token import TokenEmbedding
from models.emb_modules.embedding.position import PositionalEmbedding

from abc import *


class BaseModel(nn.Module, metaclass=ABCMeta):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.init_range = 0.01
        self.num_items = args.num_items
        vocab_size = self.num_items + 2
        self.hidden = args.hidden_units
        self.dev = args.device
        self.max_len_s = args.max_len_s
        # embedding for BERT, sum of positional, segment, token embeddings
        
        self.token = TokenEmbedding(vocab_size=vocab_size, embed_size=self.hidden)
        self.max_len = args.max_len
        #self.position_s = PositionalEmbedding(max_len=self.max_len_s, d_model=self.hidden)
        
        self.position_t = PositionalEmbedding(max_len=self.max_len, d_model=self.hidden)
        #self.decoder = decoder(self.hidden,self.num_items,self.token,self.dev)
        
    
    def init_weights(self, module):

        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=self.init_range)
 
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()



    @classmethod
    @abstractmethod
    def code(cls):
        pass

