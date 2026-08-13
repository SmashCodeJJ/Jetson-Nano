#!/usr/bin/env python3
"""
CPU vs GPU (MPS/CUDA) benchmark for ResNet50 inference.
Replicates the Jetson Nano CPU vs GPU comparison on available hardware.
Original Jetson results: GPU ~2x faster than CPU; TensorRT ~2x faster than GPU.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.backends.cudnn as cudnn
from torchvision import models

ROOT = Path(__file__).resolve().parent.parent
RESULTS_PATH = ROOT / "results" / "cpu_gpu_benchmark.json"


def benchmark(model, device: str, input_shape=(32, 3, 224, 224), nwarmup=10, nruns=30) -> dict:
    cudnn.benchmark = True
    input_data = torch.randn(input_shape).to(device)
    model = model.to(device)
    model.eval()

    print(f"\nBenchmarking on {device.upper()} ...")
    with torch.no_grad():
        for _ in range(nwarmup):
            model(input_data)

    if device == "cuda":
        torch.cuda.synchronize()
    elif device == "mps":
        torch.mps.synchronize()

    timings = []
    with torch.no_grad():
        for i in range(1, nruns + 1):
            start = time.time()
            model(input_data)
            if device == "cuda":
                torch.cuda.synchronize()
            elif device == "mps":
                torch.mps.synchronize()
            timings.append(time.time() - start)
            if i % 10 == 0:
                print(f"  Iteration {i}/{nruns}, avg batch time {np.mean(timings) * 1000:.2f} ms")

    avg_ms = float(np.mean(timings) * 1000)
    print(f"Average batch time on {device}: {avg_ms:.2f} ms")
    return {"device": device, "avg_batch_ms": avg_ms, "batch_size": input_shape[0]}


def main() -> int:
    print("Loading ResNet50 (pretrained)...")
    model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)

    results = {"platform": sys.platform, "benchmarks": [], "jetson_reference": {
        "gpu_fps_yolov5": 6.5,
        "tensorrt_fps_yolov5": 13.35,
        "note": "Original YOLOv5 FPS measured on Jetson Nano with Jetpack 4.5"
    }}

    cpu_result = benchmark(model, "cpu")
    results["benchmarks"].append(cpu_result)

    if torch.cuda.is_available():
        results["benchmarks"].append(benchmark(model, "cuda"))
    elif torch.backends.mps.is_available():
        results["benchmarks"].append(benchmark(model, "mps"))
    else:
        print("No CUDA/MPS available; CPU-only benchmark recorded.")

    if len(results["benchmarks"]) >= 2:
        cpu_ms = results["benchmarks"][0]["avg_batch_ms"]
        accel_ms = results["benchmarks"][1]["avg_batch_ms"]
        speedup = cpu_ms / accel_ms
        results["speedup"] = round(speedup, 2)
        print(f"\nAccelerator speedup vs CPU: {speedup:.2f}x")

    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with RESULTS_PATH.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"Saved results to {RESULTS_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
