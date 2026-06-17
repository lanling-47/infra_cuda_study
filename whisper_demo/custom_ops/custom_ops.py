"""
Custom CUDA operations for Whisper optimization (V2 - Batched)
Flash Attention, LayerNorm, Softmax
"""

import torch
import torch.nn as nn

try:
    import custom_cuda_ops
    CUSTOM_OPS_AVAILABLE = True
except ImportError:
    CUSTOM_OPS_AVAILABLE = False
    print("Warning: custom_cuda_ops not found. Run 'python setup.py install' to build.")


# ============================================================================
# Batched APIs (for Whisper model surgery)
# ============================================================================

def flash_attention_batched(Q, K, V, scale, causal=False):
    """
    Batched Flash Attention for multi-head attention
    Q, K, V: [B, H, N, D]
    scale: attention scale factor (typically 1/sqrt(D))
    causal: whether to apply causal masking
    Returns: [B, H, N, D]
    """
    if not CUSTOM_OPS_AVAILABLE:
        # Fallback to PyTorch SDPA
        return torch.nn.functional.scaled_dot_product_attention(Q, K, V, is_causal=causal, scale=scale)
    return custom_cuda_ops.flash_attention_batched_forward(Q, K, V, scale, causal)


def layernorm_3d(x, weight, bias, eps=1e-5):
    """
    3D LayerNorm for [B, N, D] tensors
    """
    if not CUSTOM_OPS_AVAILABLE:
        return nn.functional.layer_norm(x, weight.shape, weight, bias, eps)
    return custom_cuda_ops.layernorm_3d_forward(x, weight, bias, eps)


def softmax_4d(x):
    """
    4D Softmax for [B, H, N, M] attention weights
    """
    if not CUSTOM_OPS_AVAILABLE:
        return torch.softmax(x, dim=-1)
    return custom_cuda_ops.softmax_3d_forward(x)


# ============================================================================
# Original 2D APIs (backward compatibility)
# ============================================================================

class FlashAttention(nn.Module):
    """Flash Attention with online softmax"""

    def __init__(self, causal=False):
        super().__init__()
        self.causal = causal

    def forward(self, Q, K, V):
        if not CUSTOM_OPS_AVAILABLE:
            scale = 1.0 / (Q.size(-1) ** 0.5)
            attn = torch.matmul(Q, K.transpose(-2, -1)) * scale
            if self.causal:
                mask = torch.triu(torch.ones(Q.size(0), K.size(0), device=Q.device), diagonal=1).bool()
                attn.masked_fill_(mask, float('-inf'))
            attn = torch.softmax(attn, dim=-1)
            return torch.matmul(attn, V)

        return custom_cuda_ops.flash_attention_forward(Q, K, V, self.causal)


class LayerNorm(nn.Module):
    """LayerNorm with Welford's algorithm"""

    def __init__(self, normalized_shape, eps=1e-5):
        super().__init__()
        if isinstance(normalized_shape, int):
            normalized_shape = (normalized_shape,)
        self.normalized_shape = normalized_shape
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))

    def forward(self, x):
        if not CUSTOM_OPS_AVAILABLE:
            return nn.functional.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        return custom_cuda_ops.layernorm_forward(x, self.weight, self.bias, self.eps)


class Softmax(nn.Module):
    """Online Softmax"""

    def __init__(self, dim=-1):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        if not CUSTOM_OPS_AVAILABLE:
            return torch.softmax(x, dim=self.dim)
        if x.dim() != 2 or self.dim != -1:
            return torch.softmax(x, dim=self.dim)
        return custom_cuda_ops.softmax_forward(x)


def flash_attention(Q, K, V, causal=False):
    return FlashAttention(causal=causal)(Q, K, V)

def layer_norm(x, weight, bias, eps=1e-5):
    if not CUSTOM_OPS_AVAILABLE:
        return nn.functional.layer_norm(x, weight.shape, weight, bias, eps)
    return custom_cuda_ops.layernorm_forward(x, weight, bias, eps)

def softmax(x, dim=-1):
    if not CUSTOM_OPS_AVAILABLE or x.dim() != 2 or dim != -1:
        return torch.softmax(x, dim=dim)
    return custom_cuda_ops.softmax_forward(x)
