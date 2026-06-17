"""
Model surgery: Selectively replace Whisper's operations with custom CUDA kernels

Key insight from micro-benchmarks:
- LayerNorm: custom kernel is ~2x faster -> REPLACE
- Flash Attention: PyTorch SDPA is 3-20x faster -> KEEP SDPA
- Softmax: similar performance -> use in fallback path

Strategy: Replace only LayerNorm, keep SDPA for attention
"""
import torch
import torch.nn.functional as F
import whisper
from whisper.model import MultiHeadAttention, LayerNorm, Linear

try:
    import custom_cuda_ops
    CUSTOM_KERNELS_AVAILABLE = True
    print("✓ Custom CUDA kernels loaded")
except ImportError:
    CUSTOM_KERNELS_AVAILABLE = False
    print("⚠ Custom CUDA kernels not available")


# ============================================================================
# Version 1: LayerNorm only (safest, proven speedup)
# ============================================================================

def patched_layernorm_forward(self, x):
    """
    Replace LayerNorm.forward with custom kernel
    Input shape: [batch, seq, hidden_dim] = [1, N, 384]

    Supports FP16 directly (no conversion needed!)
    """
    # Call custom kernel directly (supports FP16)
    out = custom_cuda_ops.layernorm_3d_forward(x, self.weight, self.bias, self.eps)
    return out


def patch_whisper_layernorm(model):
    """Replace only LayerNorm with custom kernel (proven 2x speedup)"""
    if not CUSTOM_KERNELS_AVAILABLE:
        print("Cannot patch: custom kernels not available")
        return model

    LayerNorm.forward = patched_layernorm_forward
    print("✓ Patched LayerNorm.forward (custom kernel, ~2x speedup)")

    # Keep SDPA enabled (it's faster than our custom Flash Attention)
    MultiHeadAttention.use_sdpa = True
    print("✓ Keeping PyTorch SDPA enabled (3-20x faster than custom kernel)")

    return model


# ============================================================================
# Version 2: LayerNorm + custom Flash Attention (for comparison)
# ============================================================================

def patched_qkv_attention_custom(self, q, k, v, mask=None):
    """Use custom Flash Attention kernel"""
    n_batch, n_ctx, n_state = q.shape
    d_head = n_state // self.n_head
    scale = d_head ** -0.5

    q = q.view(n_batch, n_ctx, self.n_head, d_head).permute(0, 2, 1, 3).contiguous()
    k = k.view(n_batch, n_ctx, self.n_head, d_head).permute(0, 2, 1, 3).contiguous()
    v = v.view(n_batch, n_ctx, self.n_head, d_head).permute(0, 2, 1, 3).contiguous()

    is_causal = mask is not None and n_ctx > 1
    out = custom_cuda_ops.flash_attention_batched_forward(
        q.float(), k.float(), v.float(), scale, is_causal
    )

    out = out.permute(0, 2, 1, 3).flatten(start_dim=2).to(q.dtype)
    return out, None


def patch_whisper_full(model):
    """Replace both LayerNorm and Attention with custom kernels"""
    if not CUSTOM_KERNELS_AVAILABLE:
        print("Cannot patch: custom kernels not available")
        return model

    LayerNorm.forward = patched_layernorm_forward
    MultiHeadAttention.qkv_attention = patched_qkv_attention_custom
    MultiHeadAttention.use_sdpa = False

    print("✓ Patched LayerNorm.forward + MultiHeadAttention.qkv_attention")
    print("  Note: Custom Flash Attention is slower than SDPA, use 'layernorm' mode instead")

    return model


# ============================================================================
# Version 3: Kernel-level profiling
# ============================================================================

def profile_kernels(model, audio, n_iters=5):
    """Profile individual kernel time contribution"""
    import time

    # First run with original
    print("\n--- Original (SDPA + PyTorch LayerNorm) ---")
    MultiHeadAttention.use_sdpa = True
    def original_ln_forward(self, x):
        return super(LayerNorm, self).forward(x.float()).type(x.dtype)
    LayerNorm.forward = original_ln_forward

    # Warmup
    for _ in range(3):
        _ = model.transcribe(audio, language="en", fp16=True)
        torch.cuda.synchronize()

    latencies = []
    for _ in range(n_iters):
        torch.cuda.synchronize()
        start = time.time()
        _ = model.transcribe(audio, language="en", fp16=True)
        torch.cuda.synchronize()
        latencies.append((time.time() - start) * 1000)

    print(f"  Avg: {sum(latencies)/len(latencies):.2f} ms")

    # Now with custom LayerNorm
    print("\n--- Custom LayerNorm + SDPA ---")
    LayerNorm.forward = patched_layernorm_forward

    for _ in range(3):
        _ = model.transcribe(audio, language="en", fp16=True)
        torch.cuda.synchronize()

    latencies2 = []
    for _ in range(n_iters):
        torch.cuda.synchronize()
        start = time.time()
        _ = model.transcribe(audio, language="en", fp16=True)
        torch.cuda.synchronize()
        latencies2.append((time.time() - start) * 1000)

    print(f"  Avg: {sum(latencies2)/len(latencies2):.2f} ms")

    improvement = (sum(latencies) - sum(latencies2)) / sum(latencies) * 100
    print(f"\n  Improvement: {improvement:+.1f}%")

    return latencies, latencies2


def unpatch_whisper_model():
    """Restore original Whisper model methods"""
    def original_qkv_attention(self, q, k, v, mask=None):
        n_batch, n_ctx, n_state = q.shape
        scale = (n_state // self.n_head) ** -0.25
        q = q.view(*q.shape[:2], self.n_head, -1).permute(0, 2, 1, 3)
        k = k.view(*k.shape[:2], self.n_head, -1).permute(0, 2, 1, 3)
        v = v.view(*v.shape[:2], self.n_head, -1).permute(0, 2, 1, 3)

        if MultiHeadAttention.use_sdpa:
            from torch.nn.functional import scaled_dot_product_attention
            a = scaled_dot_product_attention(q, k, v, is_causal=mask is not None and n_ctx > 1)
            out = a.permute(0, 2, 1, 3).flatten(start_dim=2)
            return out, None
        else:
            qk = (q * scale) @ (k * scale).transpose(-1, -2)
            if mask is not None:
                qk = qk + mask[:n_ctx, :n_ctx]
            qk = qk.float()
            w = F.softmax(qk, dim=-1).to(q.dtype)
            out = (w @ v).permute(0, 2, 1, 3).flatten(start_dim=2)
            return out, qk.detach()

    MultiHeadAttention.qkv_attention = original_qkv_attention

    def original_layernorm_forward(self, x):
        return super(LayerNorm, self).forward(x.float()).type(x.dtype)
    LayerNorm.forward = original_layernorm_forward

    MultiHeadAttention.use_sdpa = True
    print("✓ Restored original methods")


if __name__ == "__main__":
    import numpy as np

    print("=" * 60)
    print("Model Surgery Test")
    print("=" * 60)

    print("\n[1/3] Loading Whisper-tiny model...")
    model = whisper.load_model("tiny", device="cuda")
    model.eval()

    print("\n[2/3] Applying LayerNorm-only patch...")
    model = patch_whisper_layernorm(model)

    print("\n[3/3] Testing inference...")
    audio = np.random.randn(16000 * 10).astype(np.float32) * 0.1

    with torch.no_grad():
        result = model.transcribe(audio, language="en", fp16=True)
        print(f"✓ Inference successful")

    # Profile
    if CUSTOM_KERNELS_AVAILABLE:
        unpatch_whisper_model()
        model = whisper.load_model("tiny", device="cuda")
        model.eval()
        profile_kernels(model, audio, n_iters=5)
