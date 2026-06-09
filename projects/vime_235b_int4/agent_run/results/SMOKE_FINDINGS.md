# 235B INT4 Smoke Test — Findings

## Status: PARTIAL PASS (3/4 phases verified)

### Verified ✅
1. **INT4 symmetric checkpoint** (125 GB): converted from BF16 HF, vLLM loads correctly
2. **vLLM 235B INT4 rollout**: 4 engines × 14s init (enforce-eager), `sleep freed 161.81 GiB` (level=2)
3. **Weight sync (Megatron→vLLM)**: 53s via HfWeightIteratorDirect (EP-aware raw mode)
4. **Rollout**: completes in ~18 min (128 samples @ 28 tok/s per GPU, enforce-eager)

### Blocked ❌
- **BF16 training OOM**: 172 GB PyTorch allocated on 184 GB GB200
  - Root cause: BF16 235B training with precision-aware optimizer needs more GPU memory than available on 4-node (16-GPU) config
  - Fix: ≥8 nodes (32 GPUs) for comfortable BF16 235B training
  - Alternatively: reduce to INT8 master weights or FP16 training

## Code Changes Made (workspace)
- `actor.py:137`: switch megatron_to_hf_mode bridge→raw AFTER loading (uses EP-aware weight sync)
- `actor.py:110`: convert_to_global_name=True always (raw-mode iterator needs global names)
- `actor.py:152`: use weight_updater_args copy to avoid mode switch affecting TensorBackuper
- `utils/arguments.py:222`: --rollout-hf-checkpoint arg (separate vLLM model from Megatron hf_checkpoint)
- `vllm_engine.py:1072`: rollout_hf_checkpoint takes priority over hf_checkpoint for vLLM

## Harness Configuration
- TP4/PP4/CP1/EP4 (ETP×EP×PP=16 constraint)
- --hf-checkpoint BF16 (for bridge loading), --rollout-hf-checkpoint INT4 (for vLLM)  
- --vllm-enforce-eager (no CUDA graphs, saves ~80 GB but slower inference)
- --global-batch-size 8, --max-tokens-per-gpu 4096
- --train-memory-margin-bytes 0

## Next Steps for Full Pass
1. Use 8+ nodes (32+ GPUs) → each GPU holds 7.3B params instead of 14.7B
2. OR: research why BF16 bridge loading allocates 172 GB (> expected ~58 GB)
3. OR: accept BF16 training OOM as known limitation for 4-node config
