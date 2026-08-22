<div align="center">

# 🐳 Qwen3.8-27B · Escha W2 · Docker

**Run a full 27B-parameter reasoning model on a single RTX 3090 or 4090, locally, with an OpenAI-compatible API.**

[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![CUDA 12.8](https://img.shields.io/badge/CUDA-12.8-76b900?logo=nvidia&logoColor=white)](https://developer.nvidia.com/cuda-toolkit)
[![Python 3.12](https://img.shields.io/badge/Python-3.12-3776ab?logo=python&logoColor=white)](https://www.python.org/)
[![Runtime](https://img.shields.io/badge/Runtime-Escha%20SGLang%20v1.2.0-blueviolet)](https://huggingface.co/EschaLabs/escha-runtime-qwen3dense)
[![Model](https://img.shields.io/badge/🤗%20Model-EschaLabs%2FQwen3.8--27B--Escha--W2-yellow)](https://huggingface.co/EschaLabs/Qwen3.8-27B-Escha-W2)
[![vimuttilabs.com](https://img.shields.io/badge/by-vimuttilabs.com-0ea5e9)](https://vimuttilabs.com)

</div>

---

## ✨ What is this?

[**Qwen3.8-27B**](https://huggingface.co/Qwen/Qwen3-27B) is the latest dense 27B reasoning model from Alibaba's Qwen team — widely praised for matching or beating models 2× its size on coding, math, and instruction-following benchmarks.

[**EschaLabs**](https://huggingface.co/EschaLabs) compressed it to a **~2.47-bit mixed-rate W2 quantization** using their proprietary `escha` format and a bespoke SGLang fork with custom CUDA kernels. The result:

- ✅ Fits on a **single 24 GB GPU** (RTX 3090 / 4090) — including the full KV-cache
- ✅ Achieves **~100 % of FP8 benchmark scores** on average
- ✅ Exposes an **OpenAI-compatible REST API** on port 8080
- ✅ Supports up to **128 k token context** (this compose config)
- ✅ Built-in **thinking / reasoning mode** via `chat_template_kwargs`
- ✅ Experimental **tensor parallelism** support (v1.2.0+, multi-GPU)

This repository gives you a **zero-fuss Docker setup** to spin it up in minutes.

---

## 🖥️ Requirements

| Component | Minimum |
|-----------|---------|
| GPU | NVIDIA RTX 3090 / 4090 (or equivalent, 24 GB VRAM) |
| CUDA driver | **≥ 12.8** |
| Docker | **≥ 26.0** with the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html) |
| Disk space | ~9 GB for model weights |
| RAM | ≥ 32 GB recommended |

> **Note:** The container image pulls `nvidia/cuda:12.8.0-devel-ubuntu24.04` and installs PyTorch 2.9 + the Escha runtime wheel at build time. Expect a 20–30 min first build.

---

## 🚀 Quick Start

### 1 · Clone this repo

```bash
git clone https://github.com/vimuttilabs/escha-qwen3.8-27b-w2-docker.git
cd escha-qwen3.8-27b-w2-docker
```

### 2 · Download the model weights

```bash
pip install huggingface_hub[cli]

# If the repo is gated, authenticate first:
huggingface-cli login

huggingface-cli download EschaLabs/Qwen3.8-27B-Escha-W2 \
    --local-dir ./models/Qwen3.8-27B-Escha-W2
```

### 3 · Configure your environment

```bash
cp .env.example .env
# Edit .env if you need to change any defaults (see reference below).
```

### 4 · Build and run

```bash
docker compose up --build
```

The server is ready when you see:

```
[SGLang] Server started at http://0.0.0.0:30000
```

### 5 · Chat!

```bash
curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen3.8-27B-Escha-W2",
    "messages": [{"role": "user", "content": "Explain the Rust borrow checker to me like I am 5."}],
    "max_tokens": 512
  }'
```

Or point any OpenAI-compatible client at `http://localhost:8080`.

---

## ⚙️ Environment Variable Reference

All variables are set in your `.env` file (copied from `.env.example`).

| Variable | Default | Description |
|----------|---------|-------------|
| `HF_TOKEN` | _(empty)_ | HuggingFace access token — required only if the model repo is gated |
| `MODEL` | `/models/Qwen3.8-27B-Escha-W2` | Path to model weights **inside** the container |
| `SERVED_NAME` | `Qwen3.8-27B-Escha-W2` | Model name reported by the API |
| `HOST` | `0.0.0.0` | Bind address inside the container |
| `ESCHA_ROUTE` | `blackwell` | Kernel launch route. `blackwell` forces the faster batch-1 decode kernel — **1.72× faster single-stream on RTX 3090** (40.7 vs 23.6 tok/s) vs the default `lovelace` route. Safe to set on Ampere/Ada/Hopper; bit-identical output. |
| `MEM` | `0.88` | Fraction of GPU VRAM to reserve (0.0–1.0) |
| `CTXLEN` | `131072` | Max context in tokens. Upstream default is 65536; this compose sets **128 k**. Requires `MEM=0.88` and `INT8=on`. |
| `MAXREQ` | `1` | Max concurrent requests. Tied to the Mamba recurrent-state pool — raise together with `MAXMAMBA` and `MEM`. |
| `MAXMAMBA` | `2` | Sizes the Mamba recurrent-state pool. Clamps `max_running_requests` on 24 GB cards; raise with `MAXREQ`. |
| `CHUNK` | `2048` | Prefill chunk size in tokens. Larger = faster prefill, higher peak VRAM. |
| `INT8` | `on` | INT8 KV-cache — saves ~20 % VRAM at negligible quality cost. Required for 128 k context on 24 GB. |

> **Tip:** The API is exposed on **host port `8080`** → container port `30000`. Change the mapping in `docker-compose.yml` if you need a different host port.

---

## 🧠 Thinking / Reasoning Mode

This is a **reasoning model** — thinking is enabled by default. The model's thought process arrives in `reasoning_content` and the final answer in `content`. Make sure your client reads both fields.

Control thinking per-request via `chat_template_kwargs` (a top-level `enable_thinking` field is silently ignored by the runtime):

```bash
curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen3.8-27B-Escha-W2",
    "messages": [{"role": "user", "content": "Prove that sqrt(2) is irrational."}],
    "chat_template_kwargs": {"enable_thinking": true, "reasoning_effort": "xhigh"}
  }'
```

| `reasoning_effort` | Behaviour |
|--------------------|----------|
| `xhigh` (default) | Asks model to validate assumptions and weigh alternatives |
| `medium` | **No instruction injected** — neutral/unsteered model |
| `low` | Asks model to keep reasoning brief |

> **Warning:** Without a thinking budget, long reasoning chains can hit `finish_reason: "length"` with `content: null`. For benchmarks or agents, set `max_tokens` generously or use the [thinking_budget](https://huggingface.co/EschaLabs/escha-runtime-qwen3dense/blob/main/sglang/thinking_budget.py) utility.

---

## 📡 API Endpoints

The server is fully OpenAI API-compatible:

| Endpoint | Description |
|----------|-------------|
| `GET  /v1/models` | List available models |
| `POST /v1/chat/completions` | Chat completions (streaming supported) |
| `POST /v1/completions` | Text completions |

**Streaming example:**

```bash
curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen3.8-27B-Escha-W2",
    "messages": [{"role": "user", "content": "Write a merge sort in Python."}],
    "stream": true
  }'
```

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────┐
│  docker-compose.yml                                     │
│                                                         │
│  ┌─────────────────────────────────────────────────┐   │
│  │  escha-sglang (container)                        │   │
│  │                                                  │   │
│  │  nvidia/cuda:12.8.0-devel-ubuntu24.04            │   │
│  │  + Python 3.12 venv                              │   │
│  │  + PyTorch 2.9 (CUDA 12.8 wheel)                │   │
│  │  + escha-runtime-qwen3dense wheel                │   │
│  │    └─ custom SGLang fork                         │   │
│  │    └─ Escha CUDA decompression kernels           │   │
│  │                                                  │   │
│  │  serve.sh  ──►  OpenAI API @ :30000              │   │
│  └──────────────────────┬──────────────────────────┘   │
│                          │ port map 8080:30000           │
└─────────────────────────┼───────────────────────────────┘
                           │
              host: http://localhost:8080
```

**Volume mounts:**

| Host path | Container path | Purpose |
|-----------|---------------|---------|
| `./models` | `/models` | Model weight files |
| `~/.cache/huggingface` | `/root/.cache/huggingface` | HF cache (tokeniser etc.) |

---

## 🧩 Integrations

### Python (`openai` library)

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8080/v1", api_key="none")

response = client.chat.completions.create(
    model="Qwen3.8-27B-Escha-W2",
    messages=[{"role": "user", "content": "What is the P vs NP problem?"}],
)
print(response.choices[0].message.content)
```

### LangChain

```python
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(
    base_url="http://localhost:8080/v1",
    api_key="none",
    model="Qwen3.8-27B-Escha-W2",
)
print(llm.invoke("Summarise the Attention Is All You Need paper.").content)
```

### Continue.dev / Open WebUI / Anything OpenAI-compatible

Point the base URL to `http://localhost:8080/v1` and the model name to `Qwen3.8-27B-Escha-W2`.

---

## 🔧 Troubleshooting

### Container exits immediately

Check that the NVIDIA Container Toolkit is installed and the GPU is visible to Docker:

```bash
docker run --rm --gpus all nvidia/cuda:12.8.0-base-ubuntu24.04 nvidia-smi
```

### `CUDA out of memory`

Lower `MEM` in your `.env` (e.g. `MEM=0.82`) or reduce `CTXLEN`.

### Build fails on the PyTorch install step

Confirm your CUDA **driver** supports CUDA 12.8 (`nvidia-smi` should show `CUDA Version: 12.x`). You can also try the stable wheel explicitly:

```bash
# In Dockerfile, replace the torch install line with:
pip install torch --index-url https://download.pytorch.org/whl/cu128
```

### Model not found

Make sure you downloaded weights into `./models/Qwen3.8-27B-Escha-W2` (relative to this repo), which maps to `/models/Qwen3.8-27B-Escha-W2` inside the container.

---

## 📦 Attribution & Licenses

| Component | Author | License |
|-----------|--------|--------|
| Qwen3-27B base model | [Alibaba Qwen Team](https://huggingface.co/Qwen) | Qwen Research License |
| Qwen3.8-27B-Escha-W2 weights | [EschaLabs](https://huggingface.co/EschaLabs) | See model card |
| escha-runtime-qwen3dense | [EschaLabs Inc.](https://eschalabs.com/) | **Apache-2.0** (no copyleft; all bundled deps are Apache/MIT/BSD-3) |
| This Docker configuration | [vimuttilabs.com](https://vimuttilabs.com) | [MIT](LICENSE) |

> **Important:** Always check the upstream model and runtime license before commercial use.

---

## 🤝 Contributing

PRs are welcome! If you've tested on additional GPU models, found better tuning parameters, or added integrations, please open an issue or PR.

---

<div align="center">

Built with ☕ by [**vimuttilabs.com**](https://vimuttilabs.com)

*Bringing the best open-source AI to your local hardware.*

</div>
