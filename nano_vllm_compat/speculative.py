"""
Speculative Decoding for nano-vllm.

Algorithm:
  For each decode step:
    1. Draft: small model generates K candidate tokens autoregressively
    2. Verify: large model processes (prompt + K candidates) in one forward pass
    3. Accept/Reject: rejection sampling against draft/target distributions
    4. Update: keep accepted tokens, resample rejected position

Key speedup:
  - Target model runs 1 forward pass for K+1 tokens (batched prefill-like)
  - Instead of K separate autoregressive steps
  - Net speedup ≈ K * (target_decode_time / verify_time) * acceptance_rate

Usage:
    from speculative import SpeculativeLLM
    llm = SpeculativeLLM(
        model_path="/root/cuda-lab/models/Qwen3-4B",       # target
        draft_model_path="/root/cuda-lab/models/Qwen3-0.6B",  # draft
        num_speculative_tokens=4,
    )
    outputs = llm.generate(prompts, sampling_params)
"""
import torch
import torch.nn.functional as F
import numpy as np
from collections import deque
from time import perf_counter
from tqdm.auto import tqdm
from transformers import AutoTokenizer
import torch.multiprocessing as mp
import atexit

from nanovllm.config import Config
from nanovllm.sampling_params import SamplingParams
from nanovllm.engine.sequence import Sequence, SequenceStatus
from nanovllm.engine.scheduler import Scheduler
from nanovllm.engine.model_runner import ModelRunner
from nanovllm.engine.block_manager import BlockManager


class SpeculativeLLM:
    """
    End-to-end speculative decoding engine.

    Uses the same ModelRunner/Scheduler infrastructure for the target model,
    and a lightweight draft runner for the draft model.
    """

    def __init__(
        self,
        model: str,
        draft_model: str,
        num_speculative_tokens: int = 4,
        enforce_eager: bool = True,
        tensor_parallel_size: int = 1,
        **kwargs,
    ):
        self.num_speculative_tokens = num_speculative_tokens

        # Build target model config and runner
        target_config = Config(model, enforce_eager=enforce_eager,
                               tensor_parallel_size=tensor_parallel_size, **kwargs)
        Sequence.block_size = target_config.kvcache_block_size
        self.target_config = target_config

        # Build draft model config
        draft_config = Config(draft_model, enforce_eager=True,
                              tensor_parallel_size=1, **kwargs)
        self.draft_config = draft_config

        # Initialize target model
        print(f"[spec] Loading target model: {model}")
        self.ps = []
        self.events = []
        if tensor_parallel_size > 1:
            ctx = mp.get_context("spawn")
            for i in range(1, tensor_parallel_size):
                event = ctx.Event()
                process = ctx.Process(target=ModelRunner,
                                       args=(target_config, i, event))
                process.start()
                self.ps.append(process)
                self.events.append(event)
        self.target_runner = ModelRunner(target_config, 0, self.events)
        self.tokenizer = AutoTokenizer.from_pretrained(model, use_fast=True)
        target_config.eos = self.tokenizer.eos_token_id
        draft_config.eos = self.tokenizer.eos_token_id

        # Initialize draft model (lightweight, no CUDA graph)
        print(f"[spec] Loading draft model: {draft_model}")
        self.draft_runner = ModelRunner(draft_config, 0, [])

        # Scheduler (shared between target and draft since they use same block manager)
        self.scheduler = Scheduler(target_config)
        atexit.register(self.exit)

    def exit(self):
        self.target_runner.call("exit")
        self.draft_runner.call("exit")
        del self.target_runner, self.draft_runner
        for p in self.ps:
            p.join()

    def add_request(self, prompt, sampling_params):
        if isinstance(prompt, str):
            prompt = self.tokenizer.encode(prompt)
        seq = Sequence(prompt, sampling_params)
        self.scheduler.add(seq)

    def _draft_step(self, seqs: list[Sequence], K: int) -> list[list[int]]:
        """
        Run draft model K times autoregressively.
        Returns K draft tokens per sequence: [[t1, t2, ..., tK], ...]
        """
        draft_tokens = [[] for _ in range(len(seqs))]

        for k in range(K):
            # Prepare decode inputs for draft model
            input_ids = []
            positions = []
            slot_mapping = []
            context_lens = []

            for seq in seqs:
                input_ids.append(seq.last_token)
                positions.append(len(seq) - 1)
                context_lens.append(len(seq))
                # Draft model uses its own block manager; for simplicity,
                # we allocate separate blocks for draft (or skip draft KV cache)
                slot_mapping.append(-1)  # skip KV cache write for draft

            input_ids_t = torch.tensor(input_ids, dtype=torch.int64,
                                        pin_memory=True).cuda(non_blocking=True)
            positions_t = torch.tensor(positions, dtype=torch.int64,
                                        pin_memory=True).cuda(non_blocking=True)
            slot_mapping_t = torch.tensor(slot_mapping, dtype=torch.int32,
                                           pin_memory=True).cuda(non_blocking=True)
            context_lens_t = torch.tensor(context_lens, dtype=torch.int32,
                                           pin_memory=True).cuda(non_blocking=True)
            block_tables_t = torch.zeros(len(seqs), 1, dtype=torch.int32).cuda()

            from nanovllm.utils.context import set_context, reset_context
            set_context(False, slot_mapping=slot_mapping_t,
                        context_lens=context_lens_t, block_tables=block_tables_t)

            logits = self.draft_runner.model.compute_logits(
                self.draft_runner.model(input_ids_t, positions_t)
            )
            reset_context()

            # Greedy sampling for draft
            next_tokens = logits.argmax(dim=-1).tolist()
            for i, tok in enumerate(next_tokens):
                draft_tokens[i].append(tok)
                seqs[i]._draft_last_token = tok

            # Update seq state for next draft step (temporary, rolled back after verify)
            for i, seq in enumerate(seqs):
                seq.last_token = next_tokens[i]

        return draft_tokens

    def _verify_step(self, seqs: list[Sequence],
                     draft_tokens: list[list[int]],
                     K: int) -> list[list[int]]:
        """
        Run target model on (prompt + K draft tokens) in one forward pass.
        Returns accepted tokens per sequence.

        Rejection sampling:
          For each position t in [0, K]:
            p_draft = draft distribution at t
            p_target = target distribution at t
            Accept with prob min(1, p_target[x] / p_draft[x])
            If rejected: sample from (p_target - p_draft)+, stop
        """
        accepted_tokens = [[] for _ in range(len(seqs))]

        for seq_idx, seq in enumerate(seqs):
            draft_seq = draft_tokens[seq_idx]
            # Build input: [seq.last_token] + draft_seq[0:K-1]
            # The target model sees positions [len(seq)-1, len(seq), ..., len(seq)+K-1]
            # and outputs logits for next-token prediction at each position.
            # logits[0] predicts draft_seq[0] (which is what seq.last_token would generate)
            # logits[K] predicts the next token after all K drafts

            # For simplicity, run target on [last_token, draft[0], ..., draft[K-1]]
            # giving K+1 logits. logits[i] is the target's prediction for position (start+i).

            # We compare: draft's prediction for position i vs target's
            # The draft model's distribution is implicit from greedy (argmax).
            # For proper rejection sampling, we need the draft's full distribution.
            # Simplification: since draft uses greedy, treat draft's top-1 as deterministic.
            # Accept if target also picks the draft token (greedy match).

            input_ids = [seq.last_token] + draft_seq[:-1] if K > 0 else [seq.last_token]
            positions = list(range(len(seq) - 1, len(seq) - 1 + len(input_ids)))

            input_ids_t = torch.tensor(input_ids, dtype=torch.int64,
                                        pin_memory=True).cuda(non_blocking=True)
            positions_t = torch.tensor(positions, dtype=torch.int64,
                                        pin_memory=True).cuda(non_blocking=True)

            # No KV cache writes for verify (we'll append accepted tokens after)
            slot_mapping_t = torch.full((len(input_ids),), -1,
                                         dtype=torch.int32).cuda()
            block_tables_t = torch.zeros(1, 1, dtype=torch.int32).cuda()
            context_lens_t = torch.tensor([len(seq)], dtype=torch.int32).cuda()

            from nanovllm.utils.context import set_context, reset_context
            set_context(True,
                        cu_seqlens_q=torch.tensor([0, len(input_ids)], dtype=torch.int32).cuda(),
                        cu_seqlens_k=torch.tensor([0, len(input_ids)], dtype=torch.int32).cuda(),
                        max_seqlen_q=len(input_ids),
                        max_seqlen_k=len(input_ids),
                        slot_mapping=slot_mapping_t,
                        context_lens=None,
                        block_tables=None)

            logits = self.target_runner.model.compute_logits(
                self.target_runner.model(input_ids_t, positions_t)
            )
            reset_context()

            # logits: [K+1, vocab_size] (for K draft tokens + 1 bonus)
            # logits[i] predicts what comes after input_ids[i]

            # Greedy accept: if target's argmax matches draft token, accept
            target_preds = logits.argmax(dim=-1).tolist()
            for i in range(K):
                if target_preds[i] == draft_seq[i]:
                    accepted_tokens[seq_idx].append(draft_seq[i])
                else:
                    # Reject: use target's prediction
                    accepted_tokens[seq_idx].append(target_preds[i])
                    break
            else:
                # All K accepted: use bonus token from target
                accepted_tokens[seq_idx].append(target_preds[K])

        return accepted_tokens

    def step(self):
        """One speculative decoding step."""
        seqs, is_prefill = self.scheduler.schedule()

        if is_prefill:
            # Prefill: normal forward pass on target model
            token_ids = self.target_runner.call("run", seqs, True)
            self.scheduler.postprocess(seqs, token_ids, True)
            outputs = [(seq.seq_id, seq.completion_token_ids)
                       for seq in seqs if seq.is_finished]
            num_tokens = sum(seq.num_scheduled_tokens for seq in seqs)
            return outputs, num_tokens

        # Decode: speculative step
        K = self.num_speculative_tokens
        orig_last_tokens = [seq.last_token for seq in seqs]
        orig_lengths = [len(seq) for seq in seqs]

        # 1. Draft: K tokens from draft model
        draft_tokens = self._draft_step(seqs, K)

        # Restore original state before verify
        for seq, orig_tok, orig_len in zip(seqs, orig_last_tokens, orig_lengths):
            seq.last_token = orig_tok
            seq.num_tokens = orig_len

        # 2. Verify: target model processes K+1 positions
        accepted = self._verify_step(seqs, draft_tokens, K)

        # 3. Apply accepted tokens
        token_ids = []
        for seq, acc_tokens in zip(seqs, accepted):
            # Append all accepted tokens (at least 1, up to K+1)
            for tok in acc_tokens:
                self.scheduler.block_manager.may_append(seq)
                seq.append_token(tok)
                seq.num_cached_tokens += 1
                if tok == self.scheduler.eos or seq.num_completion_tokens >= seq.max_tokens:
                    seq.status = SequenceStatus.FINISHED
                    self.scheduler.block_manager.deallocate(seq)
                    if seq in self.scheduler.running:
                        self.scheduler.running.remove(seq)
                    break
            token_ids.append(acc_tokens[-1])  # last accepted token

        outputs = [(seq.seq_id, seq.completion_token_ids)
                   for seq in seqs if seq.is_finished]
        total_accepted = sum(len(a) for a in accepted)
        return outputs, -total_accepted  # negative = decode tokens

    def is_finished(self):
        return self.scheduler.is_finished()

    def generate(self, prompts, sampling_params, use_tqdm=True):
        pbar = tqdm(total=len(prompts), desc="Speculative Generate",
                    dynamic_ncols=True, disable=not use_tqdm)
        if not isinstance(sampling_params, list):
            sampling_params = [sampling_params] * len(prompts)
        for prompt, sp in zip(prompts, sampling_params):
            self.add_request(prompt, sp)

        outputs = {}
        while not self.is_finished():
            output, num_tokens = self.step()
            for seq_id, token_ids in output:
                outputs[seq_id] = token_ids
                pbar.update(1)

        pbar.close()
        outputs = [outputs[seq_id] for seq_id in sorted(outputs.keys())]
        return [{"text": self.tokenizer.decode(tids), "token_ids": tids}
                for tids in outputs]
