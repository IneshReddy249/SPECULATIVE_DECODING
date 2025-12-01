#!/usr/bin/env python3
"""Minimal baseline benchmark with real GPU metrics."""
import json, time, torch, pynvml
from pathlib import Path
from tensorrt_llm.runtime import ModelRunnerCpp
from transformers import AutoTokenizer

# Config
ENGINE = "/workspace/engines_v16/target_baseline"
TOKENIZER = "/workspace/hf_models/qwen2.5-32b-instruct"
PROMPT = "Explain quantum computing in simple terms"
MAX_TOKENS, WARMUP, RUNS = 256, 2, 10

# Init
pynvml.nvmlInit()
gpu = pynvml.nvmlDeviceGetHandleByIndex(0)
tokenizer = AutoTokenizer.from_pretrained(TOKENIZER)
runner = ModelRunnerCpp.from_dir(ENGINE, rank=0, max_batch_size=1)
input_ids = tokenizer.encode(PROMPT, return_tensors="pt").int().cuda()

def run():
    torch.cuda.synchronize()
    pynvml.nvmlDeviceGetMemoryInfo(gpu)  # reset
    mem_peak, utils, powers = 0, [], []
    
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    
    out = runner.generate(input_ids, max_new_tokens=MAX_TOKENS, end_id=-1, pad_id=tokenizer.pad_token_id or 0)
    
    end.record()
    torch.cuda.synchronize()
    
    mem = pynvml.nvmlDeviceGetMemoryInfo(gpu)
    util = pynvml.nvmlDeviceGetUtilizationRates(gpu)
    power = pynvml.nvmlDeviceGetPowerUsage(gpu)
    
    tokens = out.shape[1] - input_ids.shape[1]
    latency = start.elapsed_time(end)
    return {"tokens": tokens, "latency_ms": latency, "tps": tokens*1000/latency,
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

print(f"\n=== BASELINE RESULTS ===")
print(f"Tokens: {summary['tokens']}")
print(f"Latency: {summary['latency']['avg']:.0f}ms (p50={summary['latency']['p50']:.0f}, p95={summary['latency']['p95']:.0f})")
print(f"TPS: {summary['tps']['avg']:.1f} (p50={summary['tps']['p50']:.1f}, p95={summary['tps']['p95']:.1f})")
print(f"GPU: {summary['gpu_util_avg']}% util, {summary['mem_peak_mb']}MB, {summary['power_avg_w']}W")

Path("/workspace/results").mkdir(exist_ok=True)
Path("/workspace/results/baseline.json").write_text(json.dumps(summary, indent=2))
print("\nSaved: /workspace/results/baseline.json")