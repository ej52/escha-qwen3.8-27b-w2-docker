FROM nvidia/cuda:12.8.0-devel-ubuntu24.04

# Prevent interactive prompts
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# 1. Install Python 3.12, dev headers (required for Triton JIT), and build essentials
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
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

# 2. Set up virtual environment with Python 3.12
ENV VIRTUAL_ENV=/opt/venv
RUN python3.12 -m venv $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

# 3. Upgrade pip and install the exact PyTorch 2.9.x CUDA 12.8 build
RUN pip install --upgrade pip wheel && \
    pip install "torch==2.9.*" --index-url https://download.pytorch.org/whl/cu128 && \
    pip install "huggingface_hub[cli]"

# 4. Clone the runtime repository and install the wheel + full dependency closure
RUN git clone https://huggingface.co/EschaLabs/escha-runtime-qwen3dense /workspace/runtime && \
    cd /workspace/runtime/sglang && \
    pip install ./escha-*.whl

# 5. Sanity check: Verify CUDA, Escha custom ops, and SGLang availability
RUN python -c "import torch, escha, sglang; print('[Sanity Check] torch.cuda:', torch.cuda.is_available(), '| escham_decode_gemv:', hasattr(torch.ops.escha, 'escham_decode_gemv'), '| sglang:', bool(sglang.__version__))"

WORKDIR /workspace/runtime/sglang

EXPOSE 30000

# Default entrypoint delegates to serve.sh
CMD ["bash", "serve.sh"]
