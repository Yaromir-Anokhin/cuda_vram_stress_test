# CUDA VRAM Stress Test & Data Integrity Validator

A lightweight, production-ready Python tool designed to stress-test NVIDIA GPUs. Unlike generic benchmarks that only focus on thermal load or FPS, this script focuses on **VRAM allocation limits and tensor compute data integrity**. 

It catches silent data corruption, driver crashes, and hardware instability often hidden during intense AI workloads (Stable Diffusion, LLMs) or aggressive GPU undervolting.

---

## 🔥 Key Features

* **Precise Block Allocation:** Allocates VRAM incrementally using customizable chunks to safely push the GPU to its absolute system limit.
* **Data Integrity Verification:** Performs heavy matrix multiplication across allocated blocks and continuously validates results against `NaN` or `Inf` errors.
* **Smart Performance Benchmarking:** Runs a pre-test cycle to estimate the exact time required to complete the workload.
* **Live Hardware Monitoring:** Integrates with NVIDIA Management Library (`NVML`) to track real-time temperature fluctuations and log the absolute peak temperature.
* **Built-in Performance Scoring:** Computes a specialized performance index (VRAM Throughput Index) based on locked VRAM capacity and math execution speed, perfect for hardware comparisons.
* **Automated Thermal Analytics:** Saves an elegant real-time temperature graph (`vram_temp_plot.png`) and opens it instantly upon completion.

---

## 📦 Prerequisites & Installation

Before running the script, make sure you have the official NVIDIA drivers installed, along with Python 3.8+ and the required packages:

```bash
pip install torch pynvml matplotlib
```

---

## 🚀 Usage Guide

### 1. Standard Diagnostics (Default Run)
Allocates 90% of available free VRAM in 512 MB blocks, executes a 100-cycle stress test, and automatically opens the telemetry graph window upon completion:
```bash
python cuda_vram_stress_test.py
```

### 2. Quiet Headless Testing (No GUI Popups)
Perfect for automated SSH terminals, CI/CD pipelines, or remote server background deployments. Keeps the graph file saved locally without launching an external image viewer:
```bash
python cuda_vram_stress_test.py --no-open
```

### 3. Extreme Aggressive Performance Run
Squeezes the buffer down to 98% memory capacity using tiny 32 MB chunks to maximize performance stress and benchmarking scores:
```bash
python cuda_vram_stress_test.py --ratio 0.98 --chunk-size 32 --cycles 200
```

---

## ⚙️ Available Command-Line Arguments

| Argument | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `--gpu` | `int` | `0` | Index of the target NVIDIA GPU. |
| `--ratio` | `float` | `0.90` | Fraction of available free VRAM to allocate (`0.10` - `0.98`). |
| `--chunk-size` | `int` | `512` | Memory block allocation step size in Megabytes. |
| `--cycles` | `int` | `100` | Number of matrix math validation cycles to run. Set to `0` for infinite loop. |
| `--duration` | `int` | `0` | Time limit for the test in seconds. Set to `0` for no time restriction. |
| `--log-file` | `str` | `"cuda_stress_test.log"` | Output path for the persistent logs. |
| `--plot-file` | `str` | `"vram_temp_plot.png"` | Destination path for the telemetry temperature chart. |
| `--no-open` | `bool` | `False` | Disables the automatic popup opening of the generated chart. |

---

## 📋 Sample Output Summary

```text
======================================================================
                        === FINAL TEST REPORT ===
======================================================================
Status:           SUCCESS / STABLE
Cycles Completed: 100 / 100
Total Time:       0m 21s
Max GPU Temp:     69°C
Benchmark Score:  47619 pts (VRAM Throughput Index)
----------------------------------------------------------------------
VERDICT: Hardware integrity 100% verified. No data corruption detected.
======================================================================
```

---

## 🛡️ License
This project is open-source and available under the [MIT License](LICENSE).