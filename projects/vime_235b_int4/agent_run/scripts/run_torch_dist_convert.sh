#!/bin/bash
# Convert Qwen3-235B HF → torch_dist
# No explicit TP/PP — auto-detect uses TP1/PP4 (world_size=4, 94 layers)
# Dist_checkpointing universal format allows TP4/PP2 training to load TP1/PP4 checkpoint
set -ex
echo "=== torch_dist convert START $(date -u +%FT%TZ) ==="
export PYTHONPATH=/root/vime:/root/Megatron-LM/
cd /root/vime

torchrun --nproc-per-node 4 tools/convert_hf_to_torch_dist.py \
  --disable-bias-linear --qk-layernorm --group-query-attention \
  --num-attention-heads 64 --num-query-groups 4 --kv-channels 128 \
  --num-layers 94 --hidden-size 4096 --ffn-hidden-size 12288 \
  --normalization RMSNorm --position-embedding-type rope \
  --norm-epsilon 1e-6 --rotary-percent 1.0 --swiglu \
  --untie-embeddings-and-output-weights --vocab-size 151936 \
  --rotary-base 1000000 \
  --moe-ffn-hidden-size 1536 --moe-router-score-function softmax \
  --moe-token-dispatcher-type alltoall --moe-router-topk 8 \
  --num-experts 128 --moe-grouped-gemm --moe-token-drop-policy probs \
  --moe-router-dtype fp32 --moe-permute-fusion --moe-aux-loss-coeff 0 \
  --hf-checkpoint /root/models/Qwen3-235B-A22B-Instruct-2507/ \
  --save /root/models/Qwen3-235B-A22B_torch_dist/

echo "=== torch_dist convert DONE $(date -u +%FT%TZ) ==="
