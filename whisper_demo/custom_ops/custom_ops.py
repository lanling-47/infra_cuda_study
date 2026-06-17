"""
Custom CUDA operations for Whisper optimization
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


class FlashAttention(nn.Module):
    """Flash Attention with online softmax"""

    def __init__(self, causal=False):
        super().__init__()
        self.causal = causal

    def forward(self, Q, K, V):
        """
        Args:
            Q: Query tensor [N, D]
            K: Key tensor [N, D]
            V: Value tensor [N, D]
        Returns:
            Output tensor [N, D]
        """
        if not CUSTOM_OPS_AVAILABLE:
            # Fallback to PyTorch implementation
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
        """
        Args:
            x: Input tensor [N, D]
        Returns:
            Normalized tensor [N, D]
        """
        if not CUSTOM_OPS_AVAILABLE:
            # Fallback to PyTorch implementation
            return nn.functional.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)

        return custom_cuda_ops.layernorm_forward(x, self.weight, self.bias, self.eps)


class Softmax(nn.Module):
    """Online Softmax with numerically stable computation"""

    def __init__(self, dim=-1):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        """
        Args:
            x: Input tensor [N, D]
        Returns:
            Softmax tensor [N, D]
        """
        if not CUSTOM_OPS_AVAILABLE:
            # Fallback to PyTorch implementation
            return torch.softmax(x, dim=self.dim)

        # Custom kernel expects 2D tensor with softmax over last dim
        if x.dim() != 2 or self.dim != -1:
            return torch.softmax(x, dim=self.dim)

        return custom_cuda_ops.softmax_forward(x)


def flash_attention(Q, K, V, causal=False):
    """Functional interface for Flash Attention"""
    return FlashAttention(causal=causal)(Q, K, V)


def layer_norm(x, weight, bias, eps=1e-5):
    """Functional interface for LayerNorm"""
    if not CUSTOM_OPS_AVAILABLE:
        return nn.functional.layer_norm(x, weight.shape, weight, bias, eps)
    return custom_cuda_ops.layernorm_forward(x, weight, bias, eps)


def softmax(x, dim=-1):
    """Functional interface for Softmax"""
    if not CUSTOM_OPS_AVAILABLE or x.dim() != 2 or dim != -1:
        return torch.softmax(x, dim=dim)
    return custom_cuda_ops.softmax_forward(x)
