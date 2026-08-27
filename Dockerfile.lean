# Dockerfile.lean — multi-stage lean variant for Qwen3.8-27B-Escha-W2
# Saves ~2GB vs Dockerfile (devel → runtime) + purges build-only deps.
# Per INSTALL.md § Requirements: Triton JIT needs gcc + Python headers at RUNTIME (cuda_utils.c shim at graph capture),
# so final stage keeps gcc/python3.12-dev, but drops git, git-lfs, curl, libnuma-dev, venv tooling, and devel toolkit.
# Verified on 1.2.0+qwen3dense. Build with: docker build -f Dockerfile.lean -t escha-lean .  or  docker compose -f docker-compose.yml -f docker-compose.lean.yml ...

# ---------- Stage 1: builder (full toolkit + git + LFS + wheel install) ----------
FROM nvidia/cuda:12.8.0-devel-ubuntu24.04 AS builder

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.12 \
    python3.12-venv \
    python3.12-dev \
    python3-pip \
    build-essential \
    libnuma-dev \
    libnuma1 \
    git \
    git-lfs \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/* && \
    git lfs install --skip-repo

WORKDIR /workspace

ENV VIRTUAL_ENV=/opt/venv
RUN python3.12 -m venv $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

# Torch must be cu128 2.9.x before wheel (wheel has no torch pin, ABI-linked to 2.9.x+cu12)
RUN pip install --upgrade pip wheel && \
    pip install "torch==2.9.*" --index-url https://download.pytorch.org/whl/cu128

# Clone runtime (LFS wheel) and install wheel + 61 pinned deps (transformers>=5.8, flashinfer, xgrammar...)
# Do not keep .git history in final image — copy only sglang/ without .git
RUN git clone https://huggingface.co/EschaLabs/escha-runtime-qwen3dense /workspace/runtime && \
    cd /workspace/runtime/sglang && \
    pip install ./escha-*.whl && \
    python -c "import torch, escha, sglang; print('[Builder Check] escham_decode_gemv:', hasattr(torch.ops.escha, 'escham_decode_gemv'), '| sglang:', sglang.__version__)" && \
    rm -rf /workspace/runtime/.git /workspace/runtime/.gitattributes && \
    pip cache purge 2>/dev/null || true && \
    find /opt/venv -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true


# ---------- Stage 2: runtime (minimal, retains JIT-required gcc + headers) ----------
FROM nvidia/cuda:12.8.0-runtime-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# Keep only runtime deps per INSTALL.md:
# - python3.12 + python3.12-dev + gcc (build-essential minimal) for Triton JIT (cuda_utils.c -> cc at graph capture)
# - libnuma1 (not -dev) for SGLang numa
# - ca-certificates for tokenizer TLS fallback
# Purged: git/git-lfs, curl, libnuma-dev, python3-pip/venv host, devel toolkit (ptxas ships inside triton)
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.12 \
    python3.12-dev \
    build-essential \
    libnuma1 \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

# Copy venv (torch + escha + sglang fork + deps) and runtime scripts from builder
COPY --from=builder /opt/venv /opt/venv
COPY --from=builder /workspace/runtime /workspace/runtime

ENV VIRTUAL_ENV=/opt/venv
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

# Sanity check in lean runtime (torch.cuda false at build, but escha/sglang import must succeed)
RUN python -c "import torch, escha, sglang; print('[Lean Check] escham_decode_gemv:', hasattr(torch.ops.escha, 'escham_decode_gemv'), '| sglang:', sglang.__version__)"

WORKDIR /workspace/runtime/sglang

EXPOSE 30000

CMD ["bash", "serve.sh"]
