import copy
import math
import torch
import torch.nn as nn

class MultiHeadAttention(nn.Module):
    """
    LinRec’s ELU-Norm self-attention.
    """
    def __init__(
        self,
        n_heads: int,
        hidden_size: int,
        hidden_dropout_prob: float,
        attn_dropout_prob: float,
        layer_norm_eps: float,
    ):
        super().__init__()
        if hidden_size % n_heads != 0:
            raise ValueError(f"hidden_size ({hidden_size}) not divisible by n_heads ({n_heads})")
        self.num_heads = n_heads
        self.head_dim  = hidden_size // n_heads
        self.all_head_dim = hidden_size
        self.scale = math.sqrt(self.head_dim)

        self.q_proj = nn.Linear(hidden_size, hidden_size)
        self.k_proj = nn.Linear(hidden_size, hidden_size)
        self.v_proj = nn.Linear(hidden_size, hidden_size)

        self.attn_dropout = nn.Dropout(attn_dropout_prob)
        self.out_proj     = nn.Linear(hidden_size, hidden_size)
        self.out_dropout  = nn.Dropout(hidden_dropout_prob)
        self.layer_norm   = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        self.elu         = nn.ELU()

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor = None) -> torch.Tensor:
        B, T, H = x.shape

        # 1) project
        q = self.q_proj(x).view(B, T, self.num_heads, self.head_dim).permute(0,2,1,3)
        k = self.k_proj(x).view(B, T, self.num_heads, self.head_dim).permute(0,2,1,3)
        v = self.v_proj(x).view(B, T, self.num_heads, self.head_dim).permute(0,2,1,3)

        # 2) ELU + normalize
        q_e = self.elu(q)
        k_e = self.elu(k)
        q_norm = q_e / (q_e.norm(dim=-1, keepdim=True) + 1e-6)
        k_norm = k_e / (k_e.norm(dim=-1, keepdim=True) + 1e-6)

        # 3) compute scores: [B,heads,T,T]
        scores = torch.matmul(q_norm, k_norm.transpose(-1, -2)) / self.scale

        # 4) add mask: broadcast [1,1,T,T] → [B,heads,T,T]
        if attn_mask is not None:
            scores = scores + attn_mask.unsqueeze(1)

        # 5) attention weights
        weights = torch.softmax(scores, dim=-1)
        weights = self.attn_dropout(weights)

        # 6) context: [B,heads,T, T] @ [B,heads,T,head_dim] → [B,heads,T,head_dim]
        context = torch.matmul(weights, v)

        # 7) re-assemble heads → [B, T, H]
        context = context.permute(0,2,1,3).reshape(B, T, H)

        # 8) output projection + residual + layernorm
        out = self.out_proj(context)
        out = self.out_dropout(out)
        return self.layer_norm(out + x)



class FeedForward(nn.Module):
    """
    LinRec’s two-layer feed-forward sublayer with residual + LayerNorm.
    """
    def __init__(
        self,
        hidden_size: int,
        inner_size: int,
        dropout_prob: float,
        hidden_act: str = "gelu",
        layer_norm_eps: float = 1e-12,
    ):
        super().__init__()
        self.fc1 = nn.Linear(hidden_size, inner_size)
        self.act = getattr(nn.functional, hidden_act)
        self.fc2 = nn.Linear(inner_size, hidden_size)
        self.dropout = nn.Dropout(dropout_prob)
        self.layer_norm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.fc1(x)
        h = self.act(h)
        h = self.fc2(h)
        h = self.dropout(h)
        return self.layer_norm(h + x)


class TransformerLayer(nn.Module):
    """
    One LinRec Transformer block.
    """
    def __init__(
        self,
        n_heads: int,
        hidden_size: int,
        inner_size: int,
        hidden_dropout_prob: float,
        attn_dropout_prob: float,
        hidden_act: str,
        layer_norm_eps: float,
    ):
        super().__init__()
        self.attn = MultiHeadAttention(
            n_heads, hidden_size, hidden_dropout_prob,
            attn_dropout_prob, layer_norm_eps
        )
        self.ffn = FeedForward(
            hidden_size, inner_size,
            hidden_dropout_prob, hidden_act, layer_norm_eps
        )

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor = None) -> torch.Tensor:
        x = self.attn(x, attn_mask)
        return self.ffn(x)


class TransformerEncoder(nn.Module):
    """
    Stack of LinRec TransformerLayers.
    """
    def __init__(
        self,
        n_layers: int,
        n_heads: int,
        hidden_size: int,
        inner_size: int,
        hidden_dropout_prob: float,
        attn_dropout_prob: float,
        hidden_act: str,
        layer_norm_eps: float,
    ):
        super().__init__()
        layer = TransformerLayer(
            n_heads, hidden_size, inner_size,
            hidden_dropout_prob, attn_dropout_prob,
            hidden_act, layer_norm_eps
        )
        self.layers = nn.ModuleList([copy.deepcopy(layer) for _ in range(n_layers)])

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: torch.Tensor = None,
        output_all_encoded_layers: bool = True
    ) -> list[torch.Tensor]:
        all_hidden = []
        for layer in self.layers:
            x = layer(x, attn_mask)
            if output_all_encoded_layers:
                all_hidden.append(x)
        if not output_all_encoded_layers:
            all_hidden.append(x)
        return all_hidden


class Decoder(nn.Module):
    """
    Dot-product decoder over the shared item embeddings.
    """
    def __init__(self, hidden_size: int, num_items: int, token_embed: nn.Embedding):
        super().__init__()
        self.token = token_embed
        self.vocab_size = num_items + 1

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, H]
        weight = self.token.weight[: self.vocab_size]  # [V, H]
        # logits: [B, T, V]
        return x @ weight.t()


class LinRec(nn.Module):
    """
    Stand-alone LinRec model (SASRec backbone + LinRec blocks).
    """
    @classmethod
    def code(cls) -> str:
        return "linrec"

    def __init__(self, args):
        super().__init__()
        # embeddings
        self.item_embedding     = nn.Embedding(
            args.num_items + 1, args.hidden_units, padding_idx=0
        )
        self.position_embedding = nn.Embedding(
            args.max_len, args.hidden_units
        )
        self.embed_dropout      = nn.Dropout(p=args.dropout_rate_emb)

        # transformer encoder
        self.encoder = TransformerEncoder(
            n_layers            = args.num_blocks,
            n_heads             = args.num_heads,
            hidden_size         = args.hidden_units,
            inner_size          = args.hidden_units * 4,
            hidden_dropout_prob = args.dropout_rate,
            attn_dropout_prob   = args.dropout_rate,
            hidden_act          = "gelu",
            layer_norm_eps      = args.layer_norm_eps,
        )

        # decoder
        self.decoder = Decoder(
            args.hidden_units, args.num_items, self.item_embedding
        )

        # initialize weights
        for module in self.modules():
            if isinstance(module, (nn.Linear, nn.Embedding)):
                nn.init.normal_(module.weight, mean=0.0, std=getattr(args, "initializer_range", 0.02))
            if isinstance(module, nn.LayerNorm):
                nn.init.constant_(module.bias, 0.0)
                nn.init.constant_(module.weight, 1.0)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.constant_(module.bias, 0.0)

    def forward(self, seq: torch.LongTensor):
        """
        seq: [B, T] input item IDs
        returns: (hidden_states [B, T, H], logits [B, T, V])
        """
        B, T = seq.size()
        device = seq.device

        # position ids
        pos = torch.arange(T, device=device).unsqueeze(0).expand(B, T)

        # embed + dropout
        x = self.item_embedding(seq) + self.position_embedding(pos)
        x = self.embed_dropout(x)

        # attention mask: causal + padding
        pad_mask = (seq == 0)        # [B, T]
        causal  = torch.triu(torch.ones(T, T, device=device) * float(-1e9), diagonal=1)
        attn_mask = causal.unsqueeze(0)  # [1, T, T]

        # encode
        hidden_layers = self.encoder(x, attn_mask, output_all_encoded_layers=True)
        last_hidden   = hidden_layers[-1]  # [B, T, H]

        # decode
        logits = self.decoder(last_hidden)  # [B, T, V]

        return last_hidden, logits

    def full_sort_predict(self, seq: torch.LongTensor) -> torch.Tensor:
        """
        For evaluation: returns scores over the entire item vocab at the last position.
        """
        _, logits = self.forward(seq)
        # take final step
        return logits[:, -1, :]
