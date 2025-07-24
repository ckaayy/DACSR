import torch
import torch.nn as nn
import torch.nn.functional as F
from mamba_ssm import Mamba
from typing import Optional, Tuple


class SIGMA(nn.Module):
   @classmethod
   def code(cls):
       return "SIGMA"
  
   def __init__(
       self,
       num_items,
       hidden_size: int = 64,
       num_layers: int = 1,
       dropout_prob: float = 0.2,
       loss_type: str = 'CE',
       d_state: int = 32,
       d_conv: int = 4,
       expand: int = 2,
   ):
       super(SIGMA, self).__init__()
       self.n_items = num_items
       self.hidden_size = hidden_size
       self.num_layers = num_layers
       self.dropout_prob = dropout_prob
       self.loss_type = loss_type
       self.d_state = d_state
       self.d_conv = d_conv
       self.expand = expand


       # token embedding
       self.item_embedding = nn.Embedding(self.n_items, self.hidden_size, padding_idx=0)
       self.LayerNorm = nn.LayerNorm(self.hidden_size, eps=1e-12)
       self.dropout = nn.Dropout(self.dropout_prob)


       # stacked Mamba + FFN layers
       self.mamba_layers = nn.ModuleList([
           SIGMALayer(
               d_model=self.hidden_size,
               d_state=self.d_state,
               d_conv=self.d_conv,
               expand=self.expand,
               dropout=self.dropout_prob,
           ) for _ in range(self.num_layers)
       ])


       # loss functions
       if self.loss_type not in ['BPR', 'CE']:
           raise ValueError("loss_type must be 'BPR' or 'CE'")
       if self.loss_type == 'CE':
           self.ce_loss = nn.CrossEntropyLoss()


       # initialize weights
       self._init_weights()


   def _init_weights(self):
       for m in self.modules():
           if isinstance(m, (nn.Linear, nn.Embedding)):
               m.weight.data.normal_(0.0, 0.02)
           if isinstance(m, nn.LayerNorm):
               m.bias.data.zero_()
               m.weight.data.fill_(1.0)
           if isinstance(m, nn.Linear) and m.bias is not None:
               m.bias.data.zero_()


   def forward(
       self,
       item_seq: torch.LongTensor,                    # [B, L]
       item_seq_len: Optional[torch.LongTensor] = None  # [B]
   ) -> Tuple[torch.FloatTensor, torch.FloatTensor]:   # (context, logits)
       # 1) infer lengths if not provided
       V = self.item_embedding.num_embeddings
       seq_min = int(item_seq.min().item())
       seq_max = int(item_seq.max().item())
       if seq_min < 0 or seq_max >= V:
           # print once and maybe crash so you can see the bad index
           print(f"[SIGMA.forward] OUT‐OF‐RANGE TOKEN!  min={seq_min}, max={seq_max}, vocab_size={V}")
           raise ValueError("Found token ID outside embedding range")
       # now safely clamp
       item_seq = item_seq.clamp(0, V-1)


       if item_seq_len is None:
           item_seq_len = (item_seq != 0).sum(dim=1)  # [B]


       # 2) clamp into [1, L] so later we never index out of bounds
       max_len = item_seq.size(1)
       item_seq_len = item_seq_len.clamp(min=1, max=max_len)


       # 3) embed + dropout + layernorm
       emb = self.item_embedding(item_seq)    # [B, L, D]
       emb = self.dropout(emb)
       emb = self.LayerNorm(emb)


       # 4) pass through all GMamba layers
       for layer in self.mamba_layers:
           emb = layer(emb)                   # [B, L, D]


       # 5) compute logits over your full vocabulary
       logits = emb @ self.item_embedding.weight.t()  # [B, L, V]


       # return the full sequence of contexts + logits
       return emb, logits


   def compute_logits(self, seq_output: torch.FloatTensor) -> torch.FloatTensor:
       """
       seq_output: [batch, hidden_size]
       returns: [batch, num_items]
       """
       return torch.matmul(seq_output, self.item_embedding.weight.t())


   def compute_loss(
       self,
       item_seq: torch.LongTensor,
       item_seq_len: torch.LongTensor,
       pos_items: torch.LongTensor,
       neg_items: torch.LongTensor = None,
   ) -> torch.FloatTensor:
       seq_out = self.forward(item_seq, item_seq_len)
       if self.loss_type == 'BPR':
           pos_emb = self.item_embedding(pos_items)
           neg_emb = self.item_embedding(neg_items)
           pos_score = (seq_out * pos_emb).sum(dim=-1)
           neg_score = (seq_out * neg_emb).sum(dim=-1)
           # BPR loss: -log(sigmoid(pos - neg))
           loss = -F.logsigmoid(pos_score - neg_score).mean()
           return loss
       else:
           logits = self.compute_logits(seq_out)
           return self.ce_loss(logits, pos_items)




class SIGMALayer(nn.Module):
   def __init__(
       self,
       d_model: int,
       d_state: int,
       d_conv: int,
       expand: int,
       dropout: float,
   ):
       super(SIGMALayer, self).__init__()
       self.gmamba = GMambaBlock(d_model, d_state, d_conv, expand)
       self.dropout = nn.Dropout(dropout)
       self.LayerNorm = nn.LayerNorm(d_model, eps=1e-12)
       self.ffn = FeedForward(d_model, inner_size=d_model * 4, dropout=dropout)


   def forward(self, x: torch.FloatTensor) -> torch.FloatTensor:
       out = self.gmamba(x)
       out = self.LayerNorm(self.dropout(out) + x)
       out = self.ffn(out)
       return out




class GMambaBlock(nn.Module):
   def __init__(
       self,
       d_model: int,
       d_state: int,
       d_conv: int,
       expand: int,
   ):
       super(GMambaBlock, self).__init__()
       self.combining_weights = nn.Parameter(torch.tensor([0.1, 0.1, 0.8], dtype=torch.float32))
       self.dense1 = nn.Linear(d_model, d_model)
       self.dense2 = nn.Linear(d_model, d_model)
       self.projection = nn.Linear(d_model, d_model)
       self.mamba = Mamba(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand)
       self.gru = nn.GRU(d_model, d_model, num_layers=1, bias=False, batch_first=True)
       self.selective_gate_sig = nn.Sequential(nn.Sigmoid(), nn.Linear(d_model, d_model))
       self.selective_gate_si = nn.Sequential(nn.SiLU(),   nn.Linear(d_model, d_model))
       self.selective_gate    = nn.Dropout(0.2)
       self.conv1d            = nn.Conv1d(d_model, d_model, kernel_size=3, padding=1)


   def forward(self, x: torch.FloatTensor) -> torch.FloatTensor:
       # x: [batch, seq_len, d_model]
       w = F.softmax(self.combining_weights, dim=0)


       h1 = self.dense1(x)
       g1 = self.conv1d(x.transpose(1, 2)).transpose(1, 2)


       # reversed prefix for flipped input
       flipped = x.clone()
       L = x.size(1)
       flipped[:, :max(0, L-5), :] = x[:, :max(0, L-5), :].flip(dims=[1])
       h2 = self.dense2(flipped) + flipped


       m_out   = self.mamba(x)
       m_out_f = self.mamba(flipped)


       h1 = self.selective_gate_si(h1) + self.selective_gate_sig(h1)
       h2 = self.selective_gate_si(h2) + self.selective_gate_sig(h2)


       m1 = m_out   * h1
       m2 = m_out_f * h2
       g  = self.gru(g1)[0]


       combined = w[2]*m1 + w[1]*m2 + w[0]*g
       return self.projection(combined)




class FeedForward(nn.Module):
   def __init__(
       self,
       d_model: int,
       inner_size: int,
       dropout: float = 0.2,
   ):
       super(FeedForward, self).__init__()
       self.w1 = nn.Linear(d_model, inner_size)
       self.act = nn.GELU()
       self.dropout = nn.Dropout(dropout)
       self.w2 = nn.Linear(inner_size, d_model)
       self.LayerNorm = nn.LayerNorm(d_model, eps=1e-12)


   def forward(self, x: torch.FloatTensor) -> torch.FloatTensor:
       h = self.w1(x)
       h = self.act(h)
       h = self.dropout(h)
       h = self.w2(h)
       h = self.dropout(h)
       return self.LayerNorm(h + x)





