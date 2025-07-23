import torch
import torch.nn as nn

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
        # 这里只是一个简单的Transformer解码器示例
        self.audio_proj = nn.Linear(config.audio_latent_dim, config.n_embd)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=config.n_embd,
            nhead=config.n_head,
            dim_feedforward=4*config.n_embd,
            batch_first=True
        )
        self.transformer = nn.TransformerDecoder(decoder_layer, num_layers=config.n_layer)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size)

    def forward(self, seqs, seq_types, mask=None):
        # seqs: [audio_latent, text_ids]
        audio_latent, text_ids = seqs
        # audio_latent: (B, 1, audio_latent_dim)
        # text_ids: (B, T)
        audio_emb = self.audio_proj(audio_latent)  # (B, 1, n_embd)
        # 假设text_ids已经embedding（如BertTokenizer），这里只做简单embedding
        text_emb = nn.functional.one_hot(text_ids, num_classes=self.lm_head.out_features).float() @ self.lm_head.weight  # (B, T, n_embd)
        # Transformer解码器
        tgt = text_emb
        memory = audio_emb
        out = self.transformer(tgt, memory)
        logits = self.lm_head(out)  # (B, T, vocab_size)
        # 返回和Llama一样的结构
        return [None, logits]