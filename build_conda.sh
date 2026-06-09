#!/bin/bash

set -ex

# create conda
yes '' | "${SHELL}" <(curl -L micro.mamba.pm/install.sh)
export PS1=tmp
mkdir -p /root/.cargo/
touch /root/.cargo/env
source ~/.bashrc

# The micromamba installer writes `nodefaults` into ~/.condarc as a channel
# entry, which newer micromamba versions try to fetch as a real anaconda.org
# repo (it isn't — it's a meta-tag) and time out on. Strip it.
if [ -f ~/.condarc ]; then
  sed -i '/^\s*-\s*nodefaults\s*$/d' ~/.condarc
fi

micromamba create -n vime python=3.12 pip -c conda-forge -y
micromamba activate vime
export CUDA_HOME="$CONDA_PREFIX"

# Keep these in sync with docker/Dockerfile:
#   - VLLM_IMAGE_TAG (ARG)            -> VLLM_VERSION below
#   - MEGATRON_COMMIT (ARG)             -> MEGATRON_COMMIT below
#   - PATCH_VERSION (ARG, default "latest") -> PATCH_VERSION below
export VLLM_VERSION="v0.5.12.post1"
export VLLM_COMMIT="5a15cde858ea09b77116212a39356f2fc51b8584"
export MEGATRON_COMMIT="1dcf0dafa884ad52ffb243625717a3471643e087"
export PATCH_VERSION="latest"

export BASE_DIR=${BASE_DIR:-"/root"}
cd $BASE_DIR

# install cuda 12.9 as it's the default cuda version for torch
micromamba install -n vime \
  cuda=12.9.1 \
  cuda-nvtx=12.9.79 \
  cuda-nvtx-dev=12.9.79 \
  nccl \
  -c nvidia/label/cuda-12.9.1 \
  -c nvidia \
  -c conda-forge \
  -y
micromamba install -n vime -c conda-forge cudnn -y
# vllm's editable install builds a Rust extension (vllm-grpc via
# setuptools-rust), so the conda env needs a working rustc + cargo.
micromamba install -n vime -c conda-forge rust -y

pip install cuda-python==12.9

# install vllm. The Dockerfile starts FROM vimerl/vllm:v0.5.12.post1-cu129
# which already has vllm installed with cu129-built native kernels; we have
# to install it ourselves here. Two follow-up steps clean up the cu13 spill:
#   1. force-reinstall torch / vllm-kernel / sgl-deep-gemm to their +cu129
#      wheels (pypi defaults are cu13);
#   2. uninstall the cu13 nvidia-* runtime libs vllm dragged in, then
#      reinstall the cu12 equivalents to repair the `site-packages/nvidia/*`
#      shared dirs (pip uninstall stomps libs co-owned across cu12/cu13).
if [ ! -d "$BASE_DIR/vllm" ]; then
  cd $BASE_DIR
  git clone https://github.com/sgl-project/vllm.git
fi
cd $BASE_DIR/vllm
git checkout ${VLLM_COMMIT}
pip install -e "python[all]" --extra-index-url https://download.pytorch.org/whl/cu129
pip install --force-reinstall --no-deps \
  torch==2.11.0 torchvision torchaudio==2.11.0 \
  --index-url https://download.pytorch.org/whl/cu129
pip install --force-reinstall --no-deps \
  vllm-kernel==0.4.2.post2 sgl-deep-gemm==0.1.0 \
  --index-url https://docs.vllm.ai/whl/cu129/
pip uninstall -y \
  nvidia-cublas \
  nvidia-cuda-cupti \
  nvidia-cuda-nvrtc \
  nvidia-cuda-runtime \
  nvidia-cudnn-cu13 \
  nvidia-cufft \
  nvidia-cufile \
  nvidia-curand \
  nvidia-cusolver \
  nvidia-cusparse \
  nvidia-cusparselt-cu13 \
  nvidia-nccl-cu13 \
  nvidia-nvjitlink \
  nvidia-nvshmem-cu13 \
  nvidia-nvtx \
  nvidia-cutlass-dsl-libs-cu13 \
  || true
pip install --force-reinstall --no-deps \
  nvidia-cublas-cu12 \
  nvidia-cuda-cupti-cu12 \
  nvidia-cuda-nvrtc-cu12 \
  nvidia-cuda-runtime-cu12 \
  nvidia-cudnn-cu12==9.16.0.29 \
  nvidia-cufft-cu12 \
  nvidia-cufile-cu12 \
  nvidia-curand-cu12 \
  nvidia-cusolver-cu12 \
  nvidia-cusparse-cu12 \
  nvidia-cusparselt-cu12 \
  nvidia-nccl-cu12 \
  nvidia-nvjitlink-cu12 \
  nvidia-nvshmem-cu12 \
  nvidia-nvtx-cu12 \
  --index-url https://download.pytorch.org/whl/cu129 \
  --extra-index-url https://pypi.org/simple


pip install cmake ninja

# flash attn 2 (matches Dockerfile)
# the newest version megatron supports is v2.7.4.post1
MAX_JOBS=64 pip -v install flash-attn==2.7.4.post1 --no-build-isolation

pip install git+https://github.com/ISEEKYAN/mbridge.git@89eb10887887bc74853f89a4de258c0702932a1c --no-deps
pip install flash-linear-attention==0.4.1
# FlashQLA: optional GDN backend for Qwen3.5/Qwen3-Next (--qwen-gdn-backend flashqla; requires SM90+)
pip install git+https://github.com/QwenLM/FlashQLA.git --no-build-isolation
# tilelang (matches Dockerfile)
pip install tilelang -f https://tile-ai.github.io/whl/nightly/cu128/

pip install --no-build-isolation "transformer_engine[pytorch]==2.10.0"

NVCC_APPEND_FLAGS="--threads 4" \
  pip -v install --disable-pip-version-check --no-cache-dir \
  --no-build-isolation \
  --config-settings "--build-option=--cpp_ext --cuda_ext --parallel 8" git+https://github.com/NVIDIA/apex.git@10417aceddd7d5d05d7cbf7b0fc2daad1105f8b4

TMS_CUDA_MAJOR="${TMS_CUDA_MAJOR:-$(python -c 'import torch; print(torch.version.cuda.split(".")[0])')}"
export TMS_CUDA_MAJOR
# --no-build-isolation: TMS's setup.py needs to find nvcc + headers + the
# installed torch to build its cu${TMS_CUDA_MAJOR} native hook; pip's default
# PEP 517 build venv hides them, so the wheel comes out python-only (~46KB)
# and vllm trips `Only hook_mode=preload supports pauseable CUDA Graph`
# because the preload .so was never compiled in.
pip install -v git+https://github.com/fzyzcjy/torch_memory_saver.git@a193d9dd1b877d33c64a41cfb3db9f867df2d926 \
  --no-cache-dir --force-reinstall --no-build-isolation
# matches Dockerfile (different fork/branch from older build_conda.sh)
pip install git+https://github.com/radixark/Megatron-Bridge.git@bridge --no-deps --no-build-isolation
pip install nvidia-modelopt[torch]>=0.37.0 --no-build-isolation
pip install https://github.com/zhuzilin/sgl-router/releases/download/v0.3.2-5f8d397/vllm_router-0.3.2-cp38-abi3-manylinux_2_28_x86_64.whl --force-reinstall
python -c "import vllm_router; assert 'vime' in vllm_router.__version__"

# megatron
cd $BASE_DIR
if [ ! -d "$BASE_DIR/Megatron-LM" ]; then
  git clone https://github.com/NVIDIA/Megatron-LM.git --recursive
fi
# pre-install Megatron's build deps explicitly since we use --no-build-isolation
pip install "setuptools<80.0.0" pybind11 "packaging>=24.2"
# --no-build-isolation: setup.py builds a C++ extension (megatron.core.datasets.helpers_cpp)
# that subprocess-shells `python3 -m pybind11`; without isolation pip uses the
# current env's python which already has pybind11 installed. Otherwise the ext
# is marked optional and silently skipped, which breaks GPT dataset loading.
cd $BASE_DIR/Megatron-LM && git checkout ${MEGATRON_COMMIT} && pip install -e . --no-build-isolation

# install vime and apply patches

# if vime does not exist locally, clone it
if [ ! -d "$BASE_DIR/vime" ]; then
  cd $BASE_DIR
  git clone https://github.com/THUDM/vime.git
fi
export VIME_DIR=$BASE_DIR/vime
cd $VIME_DIR
# Install vime's pure-python runtime deps first (wandb, ray, accelerate,
# transformers, etc.) from its requirements.txt, then install vime itself
# with --no-deps so pip doesn't re-resolve and stomp our pinned native libs
# (torch+cu129, vllm-kernel+cu129, ...). The Dockerfile does the same thing
# in two RUN layers (line ~71 + line ~124).
pip install -r requirements.txt
pip install -e . --no-deps

# int4_qat kernel (matches Dockerfile)
cd $VIME_DIR/vime/backends/megatron_utils/kernels/int4_qat
pip install . --no-build-isolation

# https://github.com/pytorch/pytorch/issues/168167
pip install nvidia-cudnn-cu12==9.16.0.29
pip install "numpy<2"
# kernels 0.15.x trips a ValueError("Either a revision or a version must be
# specified") on `transformers.integrations.hub_kernels` import; pin to <0.15
# so `import vllm` works at runtime.
pip install "kernels<0.15.0"

# apply patch (matches Dockerfile: --3way + fail on conflicts)
cd $BASE_DIR/vllm
if git apply --check $VIME_DIR/docker/patch/${PATCH_VERSION}/vllm.patch 2>/dev/null; then
  git update-index --refresh || true
  git apply $VIME_DIR/docker/patch/${PATCH_VERSION}/vllm.patch --3way
  if grep -R -n '^<<<<<<< ' .; then
    echo "vllm patch failed to apply cleanly. Please resolve conflicts." >&2
    exit 1
  fi
else
  echo "vllm patch already applied or not applicable, skipping"
fi
cd $BASE_DIR/Megatron-LM
if git apply --check $VIME_DIR/docker/patch/${PATCH_VERSION}/megatron.patch 2>/dev/null; then
  git update-index --refresh || true
  git apply $VIME_DIR/docker/patch/${PATCH_VERSION}/megatron.patch --3way
  if grep -R -n '^<<<<<<< ' .; then
    echo "megatron patch failed to apply cleanly. Please resolve conflicts." >&2
    exit 1
  fi
else
  echo "megatron patch already applied or not applicable, skipping"
fi
