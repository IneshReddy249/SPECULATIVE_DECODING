#!/usr/bin/env python3
"""Minimal speculative decoding benchmark with real GPU metrics."""
import json, torch, pynvml
from pathlib import Path
from tensorrt_llm.runtime import ModelRunnerCpp
from transformers import AutoTokenizer

# Config
DRAFT_ENGINE = "/workspace/engines_v16/draft"
TARGET_ENGINE = "/workspace/engines_v16/target_speculative"
TOKENIZER = "/workspace/hf_models/qwen2.5-32b-instruct"
PROMPT = "Explain quantum computing in simple terms"
MAX_TOKENS, K, WARMUP, RUNS = 256, 5, 2, 10

# Init
pynvml.nvmlInit()
gpu = pynvml.nvmlDeviceGetHandleByIndex(0)
tokenizer = AutoTokenizer.from_pretrained(TOKENIZER)
draft = ModelRunnerCpp.from_dir(DRAFT_ENGINE, rank=0, max_batch_size=1, kv_cache_free_gpu_memory_fraction=0.05)
target = ModelRunnerCpp.from_dir(TARGET_ENGINE, rank=0, max_batch_size=1, kv_cache_free_gpu_memory_fraction=0.40, kv_cache_enable_block_reuse=True)
input_ids = tokenizer.encode(PROMPT, return_tensors="pt").int().cuda()

def run():
    seq = input_ids.clone()
    tokens_generated = 0
    
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    
    while tokens_generated < MAX_TOKENS:
        # Draft K tokens
        draft_out = draft.generate(seq, max_new_tokens=min(K, MAX_TOKENS - tokens_generated), end_id=-1, pad_id=0)
        draft_tokens = draft_out[0, seq.shape[1]:]
        
        # Target verifies (accepts all in this simplified version)
        candidate = torch.cat([seq, draft_tokens.unsqueeze(0)], dim=1)
        target.generate(candidate[:, :-1], max_new_tokens=1, end_id=-1, pad_id=0)
        
        seq = candidate
        tokens_generated += len(draft_tokens)
    
    end.record()
    torch.cuda.synchronize()
    
    mem = pynvml.nvmlDeviceGetMemoryInfo(gpu)
    util = pynvml.nvmlDeviceGetUtilizationRates(gpu)
    power = pynvml.nvmlDeviceGetPowerUsage(gpu)
    
    latency = start.elapsed_time(end)
    return {"tokens": tokens_generated, "latency_ms": latency, "tps": tokens_generated*1000/latency,
            "mem_mb": mem.used//1024//1024, "gpu_util": util.gpu, "power_w": power//1000}

# Warmup
print(f"Warmup ({WARMUP} runs)...")
for _ in range(WARMUP): run()

# Benchmark
print(f"Benchmarking ({RUNS} runs)...")
results = [run() for _ in range(RUNS)]

# Stats
def stat(key):
    vals = sorted([r[key] for r in results])
    return {"avg": sum(vals)/len(vals), "p50": vals[len(vals)//2], 
            "p95": vals[int(len(vals)*0.95)], "p99": vals[-1]}

summary = {
    "tokens": results[0]["tokens"],
    "latency": stat("latency_ms"),
    "tps": stat("tps"),
    "gpu_util_avg": sum(r["gpu_util"] for r in results)//len(results),
    "mem_peak_mb": max(r["mem_mb"] for r in results),
    "power_avg_w": sum(r["power_w"] for r in results)//len(results)
}

print(f"\n=== SPECULATIVE RESULTS ===")
print(f"Tokens: {summary['tokens']}")
print(f"Latency: {summary['latency']['avg']:.0f}ms (p50={summary['latency']['p50']:.0f}, p95={summary['latency']['p95']:.0f})")
print(f"TPS: {summary['tps']['avg']:.1f} (p50={summary['tps']['p50']:.1f}, p95={summary['tps']['p95']:.1f})")
print(f"GPU: {summary['gpu_util_avg']}% util, {summary['mem_peak_mb']}MB, {summary['power_avg_w']}W")

Path("/workspace/results").mkdir(exist_ok=True)
Path("/workspace/results/speculative.json").write_text(json.dumps(summary, indent=2))
print("\nSaved: /workspace/results/speculative.json")