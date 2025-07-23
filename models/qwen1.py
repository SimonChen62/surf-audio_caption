import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=500):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)

class QwenConfig:
    def __init__(self, block_size, audio_latent_dim, vocab_size, n_layer, n_head, n_embd):
        self.block_size = block_size
        self.audio_latent_dim = audio_latent_dim
        self.vocab_size = vocab_size
        self.n_layer = n_layer
        self.n_head = n_head
        self.n_embd = n_embd

class Qwen(nn.Module):
    def __init__(self, config: QwenConfig):
        super().__init__()
        self.config = config
        self.audio_proj = nn.Linear(config.audio_latent_dim, config.n_embd)
        self.token_embedding = nn.Embedding(config.vocab_size, config.n_embd)
        self.pos_encoder = PositionalEncoding(config.n_embd)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=config.n_embd,
            nhead=config.n_head,
            dim_feedforward=4 * config.n_embd,
            batch_first=True
        )
        self.transformer = nn.TransformerDecoder(decoder_layer, num_layers=config.n_layer)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size)
        self.lm_head.weight = self.token_embedding.weight  # 权重共享

    def forward(self, seqs, seq_types, mask=None):
        audio_latent, text_ids = seqs
        audio_emb = self.audio_proj(audio_latent)  # (B, 1, n_embd)
        text_emb = self.token_embedding(text_ids)  # (B, T, n_embd)
        text_emb = self.pos_encoder(text_emb)
        out = self.transformer(tgt=text_emb, memory=audio_emb)
        logits = self.lm_head(out)
        return [None, logits]