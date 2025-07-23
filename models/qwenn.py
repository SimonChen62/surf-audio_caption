"""
Modified from llama.py with Qwen-style architecture and features
"""
import math
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union, Dict

import torch
import torch.nn as nn
from torch.nn import functional as F


@dataclass
class QwenConfig:
    block_size: int = 2048
    audio_latent_dim: int = 2048
    vocab_size: int = 151936  # Qwen vocab size
    n_layer: int = 32
    n_head: int = 32
    n_embd: int = 4096
    intermediate_size: int = 11008
    kv_channels: int = 128
    rotary_emb_base: float = 10000.0
    rotary_pct: float = 1.0
    use_dynamic_ntk: bool = True
    use_logn_attn: bool = True
    layer_norm_epsilon: float = 1e-6
    emb_dropout_prob: float = 0.0
    attn_dropout_prob: float = 0.0
    bf16: bool = False
    fp16: bool = False
    fp32: bool = True
    use_flash_attn: bool = False
    use_cache_quantization: bool = False
    use_cache_kernel: bool = False
    softmax_in_fp32: bool = False
    seq_length: int = 2048
    no_bias: bool = True


# Default Qwen configurations
qwen_configs = {
    "7B": dict(n_layer=32, n_head=32, n_embd=4096, intermediate_size=11008),
    "14B": dict(n_layer=40, n_head=40, n_embd=5120, intermediate_size=13696),
    "72B": dict(n_layer=80, n_head=64, n_embd=8192, intermediate_size=24576),
}


class Qwen(nn.Module):
    """Qwen model with Llama-style interface but Qwen architecture"""

    def __init__(self, config: QwenConfig) -> None:
        super().__init__()

        self.config = config

        # Audio to embedding
        self.a2e = nn.Linear(config.audio_latent_dim, config.n_embd)

        # Word token embedding
        self.wte = nn.Embedding(config.vocab_size, config.n_embd)

        # Dropout
        self.drop = nn.Dropout(config.emb_dropout_prob)

        # Transformer blocks
        self.blocks = nn.ModuleList(QwenBlock(config) for _ in range(config.n_layer))

        # Output layers
        self.ln_f = RMSNorm(config.n_embd, eps=config.layer_norm_epsilon)
        self.audio_head = nn.Linear(config.n_embd, config.audio_latent_dim, bias=False)
        self.text_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # Rotary embedding
        self.rotary_ndims = int(config.kv_channels * config.rotary_pct)
        dim = self.rotary_ndims if self.rotary_ndims is not None else config.kv_channels
        self.rotary_emb = RotaryEmbedding(dim, base=config.rotary_emb_base)

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        """Initialize the weights."""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, RMSNorm):
            module.scale.data.fill_(1.0)

    def get_ntk_alpha(self, true_seq_len):
        """Calculate NTK alpha for dynamic scaling"""
        context_value = math.log(true_seq_len / self.config.seq_length, 2) + 1
        ntk_alpha = 2 ** math.ceil(context_value) - 1
        ntk_alpha = max(ntk_alpha, 1)
        return ntk_alpha

    def forward(
        self, 
        seqs: List[torch.Tensor],
        seq_types: List[str],
        mask: Optional[torch.Tensor] = None,
    ) -> List[torch.Tensor]:
        """Forward pass with Qwen-style processing

        Args:
            seqs: list of input audio embeddings or text ids
            seq_types: list of types, e.g., ["audio", "text"]
            mask: attention mask

        Returns:
            output_seqs: list of output audio embeddings or text logits
        """

        # Transform and concatenate audio embeddings and text IDs into latent
        x = self.seqs_to_latent(seqs=seqs, seq_types=seq_types)  # shape: (b, t, d)

        device = x.device
        B, T, D = x.shape

        assert T <= self.config.block_size, f"Cannot forward sequence of {T} > {self.config.block_size}"

        if mask is None:
            mask = build_causal_mask(seq_len=T).to(device)

        # Setup rotary embeddings with dynamic NTK if needed
        if self.training or not self.config.use_dynamic_ntk:
            ntk_alpha = 1.0
        else:
            ntk_alpha = self.get_ntk_alpha(T)

        rotary_pos_emb = self.rotary_emb(T, ntk_alpha=ntk_alpha)

        # Apply dropout
        x = self.drop(x)

        # Transformer blocks
        for block in self.blocks:
            x = block(x, rotary_pos_emb, mask)

        # Final layer norm
        x = self.ln_f(x)

        # Split and transform latent into audio latents and text IDs
        seq_lens = [seq.shape[1] for seq in seqs]
        output_seqs = self.latent_to_seqs(latent=x, seq_lens=seq_lens, seq_types=seq_types)

        return output_seqs

    def seqs_to_latent(
        self, 
        seqs: List[torch.Tensor], 
        seq_types: List[str]
    ) -> torch.Tensor:
        """Transform audio latents and text IDs and concatenate them into latent."""
        
        latent = []

        for seq, seq_type in zip(seqs, seq_types):
            if seq_type == "audio":
                x = self.a2e(seq)  # shape: (b, t_audio, d)
            elif seq_type == "text":
                x = self.wte(seq)  # shape: (b, t_text, d)
            else:
                raise ValueError(f"Unknown sequence type: {seq_type}")

            latent.append(x)

        latent = torch.cat(latent, dim=1)  # shape: (b, t, d)
        return latent

    def latent_to_seqs(
        self, 
        latent: torch.Tensor, 
        seq_lens: List[int], 
        seq_types: List[str]
    ) -> List[torch.Tensor]:
        """Split and transform latent into audio latents and text IDs."""

        seqs = []
        start_idx = 0

        for seq_len, seq_type in zip(seq_lens, seq_types):
            x = latent[:, start_idx : start_idx + seq_len, :]
            start_idx += seq_len

            if seq_type == "audio":
                x = self.audio_head(x)  # shape: (b, t_audio, d)
            elif seq_type == "text":
                x = self.text_head(x)  # shape: (b, t_text, vocab_size)
            else:
                raise ValueError(f"Unknown sequence type: {seq_type}")

            seqs.append(x)

        return seqs

    @torch.no_grad()
    def generate(
        self, 
        seqs: List[torch.Tensor],
        seq_types: List[str],
        max_new_tokens: int, 
        temperature: float = 1.0, 
        top_k: Optional[int] = None
    ) -> List[torch.Tensor]:
        """Auto-regressive generation with Qwen model"""

        for _ in range(max_new_tokens):
            # Forward
            outputs = self(seqs=seqs, seq_types=seq_types)

            # Text logits
            logits = outputs[-1]

            # Take the final step logits
            logits = logits[:, -1, :] / temperature  # shape: (b, vocab_size)

            # Crop the logits to only the top k options
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')

            # Convert logits to probabilities
            probs = F.softmax(logits, dim=-1)  # shape: (b, vocab_size)

            # Sample the next token
            next_token = torch.multinomial(probs, num_samples=1)  # shape: (b, 1)

            # Append the sampled token to the last seq
            seqs[-1] = torch.cat((seqs[-1], next_token), dim=1)

        return seqs


class QwenBlock(nn.Module):
    """Qwen transformer block"""

    def __init__(self, config: QwenConfig) -> None:
        super().__init__()
        self.ln_1 = RMSNorm(config.n_embd, eps=config.layer_norm_epsilon)
        self.attn = QwenAttention(config)
        self.ln_2 = RMSNorm(config.n_embd, eps=config.layer_norm_epsilon)
        self.mlp = QwenMLP(config)

    def forward(
        self,
        x: torch.Tensor,
        rotary_pos_emb: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass through Qwen block
        
        Args:
            x: (b, t, d)
            rotary_pos_emb: rotary position embeddings
            mask: attention mask

        Returns:
            x: (b, t, d)
        """
        # Pre-norm attention
        residual = x
        x = self.ln_1(x)
        x = self.attn(x, rotary_pos_emb, mask)
        x = residual + x

        # Pre-norm MLP
        residual = x
        x = self.ln_2(x)
        x = self.mlp(x)
        x = residual + x

        return x


class QwenAttention(nn.Module):
    """Qwen multi-head attention with RoPE"""

    def __init__(self, config: QwenConfig) -> None:
        super().__init__()
        
        self.hidden_size = config.n_embd
        self.num_heads = config.n_head
        self.head_dim = self.hidden_size // self.num_heads
        self.projection_size = config.kv_channels * config.n_head

        assert self.hidden_size % self.num_heads == 0

        # QKV projection
        self.c_attn = nn.Linear(config.n_embd, 3 * self.projection_size, bias=not config.no_bias)
        
        # Output projection
        self.c_proj = nn.Linear(self.projection_size, config.n_embd, bias=not config.no_bias)

        # Attention dropout
        self.attn_dropout = nn.Dropout(config.attn_dropout_prob)

        # Register causal mask
        self.register_buffer("masked_bias", torch.tensor(-1e4), persistent=False)

    def _split_heads(self, tensor, num_heads, attn_head_size):
        """Split hidden dimension into multiple heads"""
        new_shape = tensor.size()[:-1] + (num_heads, attn_head_size)
        tensor = tensor.view(new_shape)
        return tensor.permute(0, 2, 1, 3)  # (b, h, t, d_head)

    def _merge_heads(self, tensor, num_heads, attn_head_size):
        """Merge multiple heads back into hidden dimension"""
        tensor = tensor.permute(0, 2, 1, 3).contiguous()
        new_shape = tensor.size()[:-2] + (num_heads * attn_head_size,)
        return tensor.view(new_shape)

    def forward(
        self,
        x: torch.Tensor,
        rotary_pos_emb: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """Qwen attention forward pass
        
        Args:
            x: (b, t, d)
            rotary_pos_emb: rotary position embeddings
            mask: attention mask

        Returns:
            x: (b, t, d)
        """
        B, T, D = x.shape

        # QKV projection
        mixed_x_layer = self.c_attn(x)
        query, key, value = mixed_x_layer.split(self.projection_size, dim=2)

        # Split into heads
        query = self._split_heads(query, self.num_heads, self.head_dim)
        key = self._split_heads(key, self.num_heads, self.head_dim)
        value = self._split_heads(value, self.num_heads, self.head_dim)

        # Apply rotary position embedding
        query, key = apply_rotary_pos_emb(query, key, rotary_pos_emb)

        # Attention computation
        attn_output = F.scaled_dot_product_attention(
            query=query,
            key=key,
            value=value,
            attn_mask=mask,
            dropout_p=self.attn_dropout.p if self.training else 0.0,
            is_causal=True if mask is None else False
        )

        # Merge heads
        attn_output = self._merge_heads(attn_output, self.num_heads, self.head_dim)

        # Output projection
        attn_output = self.c_proj(attn_output)

        return attn_output


class QwenMLP(nn.Module):
    """Qwen MLP with SwiGLU activation"""

    def __init__(self, config: QwenConfig) -> None:
        super().__init__()
        
        # Qwen uses gate and up projections
        self.w1 = nn.Linear(config.n_embd, config.intermediate_size // 2, bias=not config.no_bias)
        self.w2 = nn.Linear(config.n_embd, config.intermediate_size // 2, bias=not config.no_bias)
        self.c_proj = nn.Linear(config.intermediate_size // 2, config.n_embd, bias=not config.no_bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """SwiGLU MLP forward pass
        
        Args:
            x: (b, t, d)
            
        Returns:
            x: (b, t, d)
        """
        a1 = self.w1(x)
        a2 = self.w2(x)
        intermediate_parallel = a1 * F.silu(a2)  # SwiGLU activation
        output = self.c_proj(intermediate_parallel)
        return output


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization (Qwen style)"""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        """RMSNorm forward pass
        
        Args:
            x: (b, t, d)
           
        Returns:
            x: (b, t, d)
        """
        norm_x = torch.mean(x ** 2, dim=-1, keepdim=True)
        output = x * torch.rsqrt(norm_x + self.eps) * self.scale
        return output


class RotaryEmbedding(nn.Module):
    """Rotary Position Embedding (RoPE) - Qwen style"""

    def __init__(self, dim, base=10000):
        super().__init__()
        self.dim = dim
        self.base = base
        
        # Cache for rotary embeddings
        self._cos_cached = None
        self._sin_cached = None
        self._seq_len_cached = 0

    def _update_cos_sin_cache(self, seq_len, device, dtype, ntk_alpha=1.0):
        """Update cached cos and sin values"""
        if seq_len != self._seq_len_cached or self._cos_cached is None or self._cos_cached.device != device:
            self._seq_len_cached = seq_len
            
            # Compute position indices
            positions = torch.arange(seq_len, device=device, dtype=dtype)
            
            # Compute frequency indices
            inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2, device=device, dtype=dtype) / self.dim))
            
            # Apply NTK scaling
            if ntk_alpha != 1.0:
                inv_freq = inv_freq / (ntk_alpha ** (self.dim / (self.dim - 2)))
            
            # Compute outer product
            freqs = torch.outer(positions, inv_freq)
            
            # Duplicate for cos and sin
            emb = torch.cat((freqs, freqs), dim=-1)
            
            self._cos_cached = emb.cos()[None, None, :, :]
            self._sin_cached = emb.sin()[None, None, :, :]

    def forward(self, seq_len, ntk_alpha=1.0):
        """Forward pass to get rotary embeddings
        
        Args:
            seq_len: sequence length
            ntk_alpha: NTK alpha scaling factor
            
        Returns:
            cos, sin: rotary embedding components
        """
        self._update_cos_sin_cache(seq_len, device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'), 
                                   dtype=torch.float32, ntk_alpha=ntk_alpha)
        
        return self._cos_cached[:, :, :seq_len, :], self._sin_cached[:, :, :seq_len, :]


def apply_rotary_pos_emb(query, key, cos_sin):
    """Apply rotary position embedding to query and key
    
    Args:
        query: (b, h, t, d_head)
        key: (b, h, t, d_head)
        cos_sin: tuple of (cos, sin) tensors
        
    Returns:
        query, key: rotated query and key tensors
    """
    cos, sin = cos_sin
    
    def rotate_half(x):
        """Rotate half the hidden dims of the input."""
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)

    query = query * cos + rotate_half(query) * sin
    key = key * cos + rotate_half(key) * sin
    
    return query, key


def build_causal_mask(seq_len: int) -> torch.Tensor:
    """Build causal attention mask"""
    ones = torch.ones((seq_len, seq_len), dtype=torch.bool)
    mask = torch.tril(ones)[None, None, :, :]  # shape: (1, 1, t, t)
    return mask
