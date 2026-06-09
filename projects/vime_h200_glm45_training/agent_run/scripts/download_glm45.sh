#!/bin/bash
# Download GLM-4.5-Air bf16 -> /mnt/lustre (long pole). No hf_transfer in env -> plain concurrent.
set -e
export HF=/home/aoshen/code/uv_envs/py312/bin/hf
exec $HF download zai-org/GLM-4.5-Air --local-dir /mnt/lustre/aoshen/glm45-air/GLM-4.5-Air
