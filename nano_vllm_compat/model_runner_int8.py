"""
Monkey-patch nano-vllm ModelRunner to apply INT8 quantization after load.

Activated by environment variable: NANOVLLM_INT8=1

Usage in test script:
    import os
    os.environ['NANOVLLM_INT8'] = '1'
    import sys
    sys.path.insert(0, '/root/cuda-lab/nano_vllm_compat')
    import model_runner_int8  # must be imported BEFORE ModelRunner is instantiated
"""
import os

if os.environ.get('NANOVLLM_INT8', '0') != '1':
    # Not enabled, do nothing
    pass
else:
    import sys
    from nanovllm.engine import model_runner as _mr

    _original_init = _mr.ModelRunner.__init__

    def _patched_init(self, config, rank, event):
        # Call original init
        _original_init(self, config, rank, event)

        # Now apply INT8 quantization to the loaded model
        import torch
        print("\n[INT8] Applying INT8 weight quantization...")

        # Add compat dir to path for int8_linear import
        _compat_dir = os.path.dirname(os.path.abspath(__file__))
        if _compat_dir not in sys.path:
            sys.path.insert(0, _compat_dir)

        from int8_quantize import quantize_model_int8

        # Quantize all linear layers
        stats = quantize_model_int8(self.model, verbose=True)

        # Re-warmup with INT8 kernels (CUDA graph capture needs this)
        print("\n[INT8] Re-warming up with INT8 kernels...")
        import torch
        from nanovllm.engine.sequence import Sequence
        from nanovllm.utils.context import set_context, reset_context

        torch.cuda.empty_cache()
        max_num_batched_tokens = config.max_num_batched_tokens
        max_model_len = config.max_model_len
        seq_len = min(max_num_batched_tokens, max_model_len)
        num_seqs = min(max_num_batched_tokens // seq_len, config.max_num_seqs)
        seqs = [Sequence([0] * seq_len) for _ in range(num_seqs)]
        for seq in seqs:
            seq.num_scheduled_tokens = seq_len
        self.run(seqs, True)
        torch.cuda.empty_cache()
        print("[INT8] Warmup complete\n")

    _mr.ModelRunner.__init__ = _patched_init
    print("[INT8] ModelRunner patched — INT8 quantization will be applied on init")
