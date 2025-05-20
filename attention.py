import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from math import sqrt


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        pe = pe.unsqueeze(0)  # Shape: (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x: [batch_size, seq_len, embedding_dim]
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


class MultiHeadAttention(nn.Module):
    def __init__(self, hidden_dim, num_heads=4):
        super().__init__()
        assert hidden_dim % num_heads == 0
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads

        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, q, k, v, mask=None):
        batch_size = q.size(0)
        
        def transform(x, proj):
            x = proj(x)  # [B, T, H]
            x = x.view(batch_size, -1, self.num_heads, self.head_dim)  # [B, T, n, h]
            return x.transpose(1, 2)  # [B, n, T, h]

        Q = transform(q, self.q_proj)
        K = transform(k, self.k_proj)
        V = transform(v, self.v_proj)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / sqrt(self.head_dim)

        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))

        attn_weights = F.softmax(scores, dim=-1)
        output = torch.matmul(attn_weights, V)

        # [B, n, T, h] -> [B, T, H]
        output = output.transpose(1, 2).contiguous().view(batch_size, -1, self.hidden_dim)
        return self.out_proj(output)


class TransformerBlock(nn.Module):
    def __init__(self, hidden_dim, num_heads, dropout=0.1):
        super().__init__()
        self.attention = MultiHeadAttention(hidden_dim, num_heads)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.ReLU(),
            nn.Linear(hidden_dim * 4, hidden_dim)
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        # Self-attention
        attn_output = self.attention(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_output))
        
        # Feed-forward
        ffn_output = self.ffn(x)
        return self.norm2(x + self.dropout(ffn_output))


class DualAttentionSeq2Seq(nn.Module):
    def __init__(
        self,
        input_vocab_size,
        output_vocab_size,
        embedding_dim=256,       # must match hidden_dim
        hidden_dim=256,
        num_layers=3,
        num_heads=8,
        dropout=0.1,
        device="cuda"
    ):
        super().__init__()
        self.device = torch.device(device)
        self.hidden_dim = hidden_dim

        self.encoder_embedding = nn.Embedding(input_vocab_size, embedding_dim)
        self.decoder_embedding = nn.Embedding(output_vocab_size, embedding_dim)
        self.pos_encoder = PositionalEncoding(embedding_dim, dropout)

        # Stack of Transformer blocks for encoder and decoder
        self.encoder_layers = nn.ModuleList([
            TransformerBlock(hidden_dim, num_heads, dropout)
            for _ in range(num_layers)
        ])
        self.decoder_layers = nn.ModuleList([
            TransformerBlock(hidden_dim, num_heads, dropout)
            for _ in range(num_layers)
        ])

        self.out_proj = nn.Linear(hidden_dim, output_vocab_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src, trg):
        # Embedding and positional encoding
        src_emb = self.pos_encoder(self.dropout(self.encoder_embedding(src)))  # [B, T, D]
        trg_emb = self.pos_encoder(self.dropout(self.decoder_embedding(trg)))  # [B, T, D]

        # Encoder
        enc_output = src_emb
        for layer in self.encoder_layers:
            enc_output = layer(enc_output)

        # Decoder
        dec_output = trg_emb
        for layer in self.decoder_layers:
            dec_output = layer(dec_output)

        # Output logits
        return self.out_proj(dec_output)  # [B, T, output_vocab_size]
