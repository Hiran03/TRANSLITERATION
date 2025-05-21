# Modified MultiHeadAttention to return attention weights

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from math import sqrt
import plotly.graph_objects as go
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

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

        self.attn_weights = None  # To store attention weights

    def forward(self, q, k, v, mask=None, return_attention=False):
        batch_size = q.size(0)

        def transform(x, proj):
            x = proj(x)
            x = x.view(batch_size, -1, self.num_heads, self.head_dim)
            return x.transpose(1, 2)

        Q = transform(q, self.q_proj)
        K = transform(k, self.k_proj)
        V = transform(v, self.v_proj)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / sqrt(self.head_dim)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))

        attn_weights = F.softmax(scores, dim=-1)
        self.attn_weights = attn_weights if return_attention else None

        output = torch.matmul(attn_weights, V)
        output = output.transpose(1, 2).contiguous().view(batch_size, -1, self.hidden_dim)
        if return_attention:
            return output, attn_weights
        return self.out_proj(output)


# In TransformerBlock, we pass return_attention to MultiHeadAttention
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

    def forward(self, x, mask=None, return_attention=False):
        attn_output = self.attention(x, x, x, mask, return_attention)
        x = self.norm1(x + self.dropout(attn_output))
        ffn_output = self.ffn(x)
        return self.norm2(x + self.dropout(ffn_output))

class TransformerDecoderBlock(nn.Module):
    def __init__(self, hidden_dim, num_heads, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(hidden_dim, num_heads)
        self.cross_attn = MultiHeadAttention(hidden_dim, num_heads)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.ReLU(),
            nn.Linear(hidden_dim * 4, hidden_dim)
        )
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.norm3 = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, enc_out, return_attention=False):
        # Self-attention (decoder)
        x = self.norm1(x + self.dropout(self.self_attn(x, x, x)))

        # Cross-attention (decoder attending to encoder)
        if return_attention:
            cross_output, attn_weights = self.cross_attn(x, enc_out, enc_out, return_attention=True)
        else:
            cross_output = self.cross_attn(x, enc_out, enc_out)
            attn_weights = None

        x = self.norm2(x + self.dropout(cross_output))
        x = self.norm3(x + self.dropout(self.ffn(x)))

        if return_attention:
            return x, attn_weights
        return x
    
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
            TransformerDecoderBlock(hidden_dim, num_heads, dropout)
            for _ in range(num_layers)
        ])

        self.out_proj = nn.Linear(hidden_dim, output_vocab_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src, trg, return_attention=False):
        src_emb = self.pos_encoder(self.dropout(self.encoder_embedding(src)))
        trg_emb = self.pos_encoder(self.dropout(self.decoder_embedding(trg)))

        # Encoder
        enc_output = src_emb
        for layer in self.encoder_layers:
            enc_output = layer(enc_output)

        # Decoder
        dec_output = trg_emb
        cross_attn_weights = None
        for i, layer in enumerate(self.decoder_layers):
            if i == 0 and return_attention:
                dec_output, cross_attn_weights = layer(dec_output, enc_output, return_attention=True)
            else:
                dec_output = layer(dec_output, enc_output)

        logits = self.out_proj(dec_output)
        if return_attention:
            return logits, cross_attn_weights  # [B, heads, T_dec, T_enc]
        return logits


plt.rcParams['font.family'] = 'Lohit Devanagari'  # or another Unicode-complete font

def plot_cross_attention_heatmaps(model, src_batch, trg_batch, idx_to_input, idx_to_output, num_plots=9):
    model.eval()
    with torch.no_grad():
        src_batch = src_batch.to(model.device)
        trg_batch = trg_batch.to(model.device)

        # Forward pass with attention
        output_logits, attn_weights = model(src_batch, trg_batch[:, :-1], return_attention=True)
        attn_weights = attn_weights.mean(dim=1)  # [B, T_dec, T_enc]

        fig, axes = plt.subplots(3, 3, figsize=(14, 12))
        axes = axes.flatten()

        for i in range(min(num_plots, src_batch.size(0))):
            ax = axes[i]
            src_tokens = [idx_to_input.get(idx.item(), '') for idx in src_batch[i]]
            trg_tokens = [idx_to_output.get(idx.item(), '') for idx in trg_batch[i, 1:]]  # Skip BOS

            heatmap = attn_weights[i, :len(trg_tokens), :len(src_tokens)].cpu().numpy()
            sns.heatmap(heatmap, xticklabels=src_tokens, yticklabels=trg_tokens, cmap="viridis", ax=ax)
            ax.set_title(f"Cross-Attn Input {i+1}")
            ax.tick_params(axis='x', rotation=90)
            ax.tick_params(axis='y', rotation=0)

        for j in range(i + 1, 9):
            fig.delaxes(axes[j])

        plt.tight_layout()
        plt.show()
        return fig
    

def interactive_dualline_attention(
    model, src_batch, trg_batch, idx_to_input, idx_to_output_deva, idx_to_output_latin, sample_idx=0
):
    model.eval()
    with torch.no_grad():
        src_batch = src_batch.to(model.device)
        trg_batch = trg_batch.to(model.device)

        output_logits, attn_weights = model(src_batch, trg_batch[:, :-1], return_attention=True)
        attn_weights = attn_weights.mean(dim=1)  # [B, T_dec, T_enc]

        src_tokens = [idx_to_input.get(idx.item(), '') for idx in src_batch[sample_idx]]
        trg_tokens_deva = [idx_to_output_deva.get(idx.item(), '') for idx in trg_batch[sample_idx, 1:]]
        trg_tokens_latin = [idx_to_output_latin.get(idx.item(), '') for idx in trg_batch[sample_idx, 1:]]

        heatmap = attn_weights[sample_idx, :len(trg_tokens_deva), :len(src_tokens)].cpu().numpy()  # [T_dec, T_enc]

        # Initial colors for source (Latin) tokens
        base_color = 'lightgray'
        src_colors = [base_color] * len(src_tokens)

        fig = go.Figure()

        # Add source Latin tokens as scatter with color
        fig.add_trace(go.Scatter(
            x=list(range(len(src_tokens))),
            y=[0] * len(src_tokens),
            text=src_tokens,
            mode='text+markers',
            textposition="top center",
            marker=dict(color=src_colors, size=16),
            textfont=dict(size=14),
            hoverinfo='text',
            showlegend=False,
            name='Input (Latin)'
        ))

        # Add target Devanagari tokens as scatter (interactive row)
        for i, (char_deva, char_latin) in enumerate(zip(trg_tokens_deva, trg_tokens_latin)):
            attention = heatmap[i]
            norm_attention = (attention - attention.min()) / (attention.max() - attention.min() + 1e-6)
            colors = [
                f'rgba(255,0,0,{score:.2f})' for score in norm_attention
            ]

            # Add a separate invisible trace for each hover interaction
            fig.add_trace(go.Scatter(
                x=list(range(len(src_tokens))),
                y=[-(i+1)] * len(src_tokens),
                mode='markers+text',
                marker=dict(color=colors, size=16),
                text=src_tokens,
                textposition="top center",
                textfont=dict(size=14),
                hoverinfo='text',
                showlegend=False,
                visible=False,
                name=f'Attention from {char_deva} ({char_latin})'
            ))

        # Add Devanagari line for interaction
        fig.add_trace(go.Scatter(
            x=list(range(len(trg_tokens_deva))),
            y=[-len(trg_tokens_deva) - 1] * len(trg_tokens_deva),
            mode='text',
            text=trg_tokens_deva,
            textposition="top center",
            textfont=dict(size=16),
            hoverinfo='text',
            showlegend=False,
            name='Target (Devanagari)'
        ))

        # Create slider steps (one for each Devanagari token)
        steps = []
        for i in range(len(trg_tokens_deva)):
            vis = [True] + [j == i + 1 for j in range(1, len(trg_tokens_deva) + 1)] + [True]
            steps.append(dict(
                method="update",
                args=[{"visible": vis}],
                label=f"{trg_tokens_deva[i]}"
            ))

        fig.update_layout(
            sliders=[dict(
                active=0,
                pad={"t": 50},
                steps=steps,
                currentvalue=dict(prefix="Focused Token: ")
            )],
            title="Cross Attention: Devanagari → Latin Input Tokens",
            xaxis=dict(showticklabels=False),
            yaxis=dict(showticklabels=False),
            width=1000,
            height=400 + len(trg_tokens_deva) * 30,
        )

        fig.show()
        return fig
