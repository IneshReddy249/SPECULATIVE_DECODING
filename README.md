# TensorRT-LLM Speculative Decoding Benchmark

> Achieved **2.26x faster inference** by implementing speculative decoding with Qwen2.5 models on NVIDIA TensorRT-LLM. This project demonstrates how pairing a small draft model (1.5B) with a large target model (32B) can significantly reduce latency while maintaining output quality.

[![TensorRT-LLM](https://img.shields.io/badge/TensorRT--LLM-v0.16.0-green)](https://github.com/NVIDIA/TensorRT-LLM)
[![CUDA](https://img.shields.io/badge/CUDA-12.x-blue)](https://developer.nvidia.com/cuda-toolkit)
[![License](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)

---

## Results

![Comparison Results](results/images/comparison.png)

| Metric | Baseline (32B) | Speculative (1.5B + 32B) | Speedup |
|--------|----------------|--------------------------|---------|
| **Latency** | 6,057 ms | 2,678 ms | **2.26x** |
| **TPS** | 42.3 | 95.6 | **2.26x** |
| **Memory** | 77 GB | 58 GB | -25% |
| GPU Util | 99% | 94% | - |
| Power | 319W | 310W | - |

### Baseline (32B only)
![Baseline Results](results/images/baseline.png)

### Speculative (1.5B + 32B)
![Speculative Results](results/images/speculative.png)

---

## Hardware

| Component | Spec |
|-----------|------|
| GPU | NVIDIA A100 80GB |
| vCPUs | 16 |
| Storage | 1TB NVMe |
| Provider | [Shadeform](https://shadeform.ai) |

---

## How It Works

**Speculative decoding** accelerates LLM inference by using two models: a fast draft model and an accurate target model.
```
BASELINE (Autoregressive - Slow):
┌─────────────────────────────────────────────────────────────────┐
│  Each token requires a full forward pass through the 32B model  │
│                                                                 │
│  Prompt → [32B] → t1 → [32B] → t2 → [32B] → t3 → [32B] → t4    │
│            24ms        24ms        24ms        24ms             │
│                                                                 │
│  4 tokens = 96ms (sequential, memory-bound)                     │
└─────────────────────────────────────────────────────────────────┘

SPECULATIVE (Draft + Verify - Fast):
┌─────────────────────────────────────────────────────────────────┐
│  Draft model generates K tokens fast, target verifies in batch  │
│                                                                 │
│  Prompt → [1.5B Draft] → [t1, t2, t3, t4, t5] → [32B Verify]   │
│              10ms              speculate           50ms         │
│                                                                 │
│  5 tokens = 60ms (parallel verification)                        │
└─────────────────────────────────────────────────────────────────┘

WHY IT'S FASTER:
- Draft model (1.5B) is ~20x faster than target (32B)
- Target verifies all K tokens in ONE forward pass (not K passes)
- If draft tokens match target's distribution → accept all
- Net result: fewer 32B forward passes = lower latency
```

---

## Models

| Model | Role | Precision | Size |
|-------|------|-----------|------|
| [Qwen2.5-1.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct) | Draft | FP16 | 3 GB |
| [Qwen2.5-32B-Instruct](https://huggingface.co/Qwen/Qwen2.5-32B-Instruct) | Target | INT8 | 33 GB |

---

## Quick Start

### 1. Clone & Download Models
```bash
git clone https://github.com/IneshReddy249/SPECULATIVE_DECODING.git
cd SPECULATIVE_DECODING

pip install huggingface_hub
huggingface-cli download Qwen/Qwen2.5-1.5B-Instruct --local-dir hf_models/qwen2.5-1.5b-instruct
huggingface-cli download Qwen/Qwen2.5-32B-Instruct --local-dir hf_models/qwen2.5-32b-instruct
```

### 2. Start Docker
```bash
docker run -it --gpus all --shm-size=16g \
    -v $(pwd):/workspace \
    nvcr.io/nvidia/tritonserver:24.12-trtllm-python-py3 bash
```

### 3. Build & Run (inside Docker)
```bash
cd /workspace
chmod +x scripts/*.sh
./scripts/convert_checkpoints.sh   # ~10 min
./scripts/build_engines.sh         # ~3 min

python3 scripts/benchmark_baseline.py
python3 scripts/benchmark_speculative.py
python3 scripts/compare.py
```

---

## Project Structure
```
SPECULATIVE_DECODING/
├── scripts/
│   ├── convert_checkpoints.sh    # Converts HF models → TensorRT-LLM checkpoints
│   ├── build_engines.sh          # Builds optimized TensorRT engines
│   ├── benchmark_baseline.py     # Benchmarks 32B model (autoregressive)
│   ├── benchmark_speculative.py  # Benchmarks 1.5B + 32B (speculative)
│   └── compare.py                # Compares results and calculates speedup
├── results/
│   └── images/
│       ├── baseline.png          # Baseline benchmark output
│       ├── speculative.png       # Speculative benchmark output
│       └── comparison.png        # Side-by-side comparison
├── hf_models/                    # HuggingFace models (gitignored)
│   ├── qwen2.5-1.5b-instruct/
│   └── qwen2.5-32b-instruct/
├── checkpoints/                  # TensorRT-LLM checkpoints (gitignored)
│   ├── draft/
│   └── target_32b/
├── engines/                      # TensorRT engines (gitignored)
│   ├── draft/
│   ├── target_baseline/
│   └── target_speculative/
├── .gitignore
└── README.md
```

---

## Metrics

All measurements are real (not estimated):

| Metric | Method |
|--------|--------|
| Latency | `torch.cuda.Event` GPU hardware timer |
| TPS | actual tokens / actual time |
| GPU Util | `pynvml` sampled during generation |
| Memory | `pynvml` peak usage |
| Power | `pynvml` average draw |

---

## References

- [TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM)
- [Speculative Decoding Paper](https://arxiv.org/abs/2211.17192)
- [Qwen2.5](https://huggingface.co/Qwen)

---

## Author

**Inesh Reddy** · [LinkedIn](https://linkedin.com/in/ineshtickoo) · [GitHub](https://github.com/IneshReddy249)

---


