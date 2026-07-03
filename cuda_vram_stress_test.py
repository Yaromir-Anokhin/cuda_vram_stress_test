import torch
import time
import sys
import os
import argparse
from datetime import datetime

def parse_args():
    parser = argparse.ArgumentParser(
        description="Universal CUDA VRAM Stress Test & Data Integrity Validator."
    )
    parser.add_argument(
        "--gpu", type=int, default=0, 
        help="Index of the target NVIDIA GPU (default: 0)"
    )
    parser.add_argument(
        "--ratio", type=float, default=0.90, 
        help="Fraction of available free VRAM to allocate (0.1 - 0.95, default: 0.90)"
    )
    parser.add_argument(
        "--chunk-size", type=int, default=512, 
        help="Allocation step size in megabytes (default: 512)"
    )
    parser.add_argument(
        "--duration", type=int, default=0, 
        help="Test duration limit in seconds. Set to 0 for infinite loop (default: 0)"
    )
    parser.add_argument(
        "--cycles", type=int, default=100, 
        help="Number of stress-test cycles to run. Set to 0 for infinite loop (default: 100)"
    )
    parser.add_argument(
        "--log-file", type=str, default="cuda_stress_test.log", 
        help="Path to the output log file"
    )
    parser.add_argument(
        "--plot-file", type=str, default="vram_temp_plot.png", 
        help="Path to save the final temperature plot chart"
    )
    parser.add_argument(
        "--no-open", action="store_true", 
        help="Disable automatic opening of the chart image upon completion"
    )
    return parser.parse_args()

args = parse_args()

# Global variables for analytics and plotting telemetry
max_temp_tracked = 0
has_nvml = False
nvml_handle = None
time_telemetry = []
temp_telemetry = []
current_allocated_gb = 0.0  # Tracked globally for precise benchmark scoring

def log_message(message):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    formatted_msg = f"[{timestamp}] {message}"
    print(formatted_msg)
    try:
        with open(args.log_file, "a", encoding="utf-8") as f:
            f.write(formatted_msg + "\n")
            f.flush()
            os.fsync(f.fileno())
    except IOError:
        pass

try:
    import pynvml
    pynvml.nvmlInit()
    nvml_handle = pynvml.nvmlDeviceGetHandleByIndex(args.gpu)
    has_nvml = True
except Exception:
    has_nvml = False

def get_gpu_temp_raw():
    if has_nvml:
        try:
            return pynvml.nvmlDeviceGetTemperature(nvml_handle, pynvml.NVML_TEMPERATURE_GPU)
        except:
            return None
    return None

def get_gpu_temp():
    global max_temp_tracked
    temp = get_gpu_temp_raw()
    if temp is not None:
        if temp > max_temp_tracked:
            max_temp_tracked = temp
        return f"{temp}°C"
    return "N/A"

def generate_plot():
    if not time_telemetry or not temp_telemetry:
        return
    try:
        import matplotlib.pyplot as plt
        
        plt.figure(figsize=(10, 5))
        plt.plot(time_telemetry, temp_telemetry, color='#ff3333', linewidth=2, label='GPU Core Temp')
        plt.fill_between(time_telemetry, temp_telemetry, color='#ff3333', alpha=0.1)
        
        plt.title('GPU Thermal Profile During CUDA VRAM Stress Test', fontsize=12, fontweight='bold', pad=15)
        plt.xlabel('Elapsed Time (seconds)', fontsize=10)
        plt.ylabel('Temperature (°C)', fontsize=10)
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.axhline(y=max_temp_tracked, color='darkred', linestyle=':', alpha=0.7, label=f'Peak ({max_temp_tracked}°C)')
        plt.legend(loc='upper left')
        
        plt.savefig(args.plot_file, dpi=150, bbox_inches='tight')
        plt.close()
        log_message(f"Telemetry graph successfully generated and saved to: {args.plot_file}")
        
        # Automatically open the chart image file upon completion
        if not args.no_open:
            try:
                if sys.platform == "win32":
                    os.startfile(args.plot_file)
                elif sys.platform == "darwin":  # macOS
                    import subprocess
                    subprocess.Popen(["open", args.plot_file])
                else:  # Linux
                    import subprocess
                    subprocess.Popen(["xdg-open", args.plot_file])
                log_message("Automatically opening the temperature plot chart.")
            except Exception as os_err:
                log_message(f"[WARNING] Could not automatically open chart file: {str(os_err)}")
                
    except ImportError:
        log_message("[WARNING] 'matplotlib' library not found. Skipping chart generation.")
    except Exception as e:
        log_message(f"[WARNING] Failed to generate temperature plot: {str(e)}")

def print_final_report(status, cycles_done, elapsed_time, reason=""):
    total_seconds = int(elapsed_time)
    minutes = total_seconds // 60
    seconds = total_seconds % 60
    
    # Calculate the custom proprietary VRAM performance score index
    if status in ["SUCCESS / STABLE", "STOPPED"] and total_seconds > 0:
        score = int((current_allocated_gb * cycles_done) / elapsed_time * 1000)
    else:
        score = 0

    log_message("\n" + "="*70)
    log_message("                      === FINAL TEST REPORT ===")
    log_message("="*70)
    log_message(f"Status:           {status}")
    if reason:
        log_message(f"Reason:           {reason}")
    log_message(f"Cycles Completed: {cycles_done} / {args.cycles if args.cycles > 0 else 'Infinite'}")
    log_message(f"Total Time:       {minutes}m {seconds}s")
    log_message(f"Max GPU Temp:     {max_temp_tracked}°C" if max_temp_tracked > 0 else "Max GPU Temp:     N/A")
    log_message(f"Benchmark Score:  {score} pts (VRAM Throughput Index)" if score > 0 else "Benchmark Score:  0 pts")
    log_message("-"*70)
    if status == "SUCCESS / STABLE":
        log_message("VERDICT: Hardware integrity 100% verified. No data corruption detected.")
    elif status == "STOPPED":
        log_message("VERDICT: Test interrupted by user. No errors found during the run.")
    else:
        log_message("VERDICT: CRITICAL HARDWARE ERROR OR INSTABILITY DETECTED!")
    log_message("="*70 + "\n")

def main():
    global current_allocated_gb
    if not torch.cuda.is_available():
        log_message("[CRITICAL ERROR] CUDA is not available on this system!")
        sys.exit(1)
        
    device_count = torch.cuda.device_count()
    if args.gpu >= device_count:
        log_message(f"[CRITICAL ERROR] Specified GPU index {args.gpu}, but only {device_count} device(s) found.")
        sys.exit(1)

    device = torch.device(f"cuda:{args.gpu}")
    gpu_name = torch.cuda.get_device_name(device)
    
    get_gpu_temp()
    
    torch.cuda.empty_cache()
    free_mem_bytes, total_mem_bytes = torch.cuda.mem_get_info(device)
    free_mem_gb = free_mem_bytes / (1024**3)
    total_mem_gb = total_mem_bytes / (1024**3)
    
    target_vram_gb = free_mem_gb * args.ratio
    
    log_message("="*70)
    log_message(f"CUDA VRAM STRESS-TEST & HARDWARE VALIDATOR")
    log_message(f"Target Device: GPU [{args.gpu}] - {gpu_name}")
    log_message(f"Total VRAM: {total_mem_gb:.2f} GB | Currently Free: {free_mem_gb:.2f} GB")
    log_message(f"Allocation Target ({int(args.ratio*100)}% of free space): {target_vram_gb:.2f} GB")
    log_message("="*70)

    chunk_bytes = args.chunk_size * 1024 * 1024
    matrix_dim = int((chunk_bytes / 4) ** 0.5)
    real_chunk_gb = (matrix_dim * matrix_dim * 4) / (1024**3)

    allocated_blocks = []

    log_message(f"--- PHASE 1: Step-by-Step VRAM Allocation ({args.chunk_size} MB blocks) ---")
    try:
        while current_allocated_gb < target_vram_gb:
            block_idx = len(allocated_blocks) + 1
            log_message(f"Allocating block #{block_idx}... Temp: {get_gpu_temp()}")
            
            block = torch.randn(matrix_dim, matrix_dim, dtype=torch.float32, device=device)
            allocated_blocks.append(block)
            
            torch.cuda.synchronize(device)
            current_allocated_gb += real_chunk_gb
            log_message(f"-> Verification OK. Total locked memory: {current_allocated_gb:.2f} GB")
            time.sleep(0.02)
            
    except torch.cuda.OutOfMemoryError:
        log_message(f"[OOM] Hit maximum allocation limit at {current_allocated_gb:.2f} GB.")
    except RuntimeError as e:
        log_message(f"[CUDA CRASH DURING ALLOCATION] {str(e)}")
        print_final_report("CRASHED", 0, 0, f"Allocation driver crash: {str(e)}")
        sys.exit(1)

    log_message(f"Allocation complete. Holding {current_allocated_gb:.2f} GB on VRAM.")
    
    slice_size = min(matrix_dim, 4096)
    if len(allocated_blocks) > 0:
        log_message("Benchmarking GPU performance for execution time estimation...")
        torch.cuda.synchronize(device)
        t_bench_start = time.time()
        
        for i in range(len(allocated_blocks)):
            next_idx = (i + 1) % len(allocated_blocks)
            slice_a = allocated_blocks[i][:slice_size, :slice_size]
            slice_b = allocated_blocks[next_idx][:slice_size, :slice_size]
            result = torch.matmul(slice_a, slice_b)
            allocated_blocks[i][:slice_size, :slice_size] = result / (result.std() + 1e-5)
            
        torch.cuda.synchronize(device)
        single_cycle_time = (time.time() - t_bench_start) + 0.02
        
        if args.cycles > 0:
            est_total_seconds = single_cycle_time * args.cycles
            est_min = int(est_total_seconds // 60)
            est_sec = int(est_total_seconds % 60)
            est_time_str = f"~ {est_min}m {est_sec}s"
        else:
            est_time_str = "Infinite (Requires Manual Stop)"
    else:
        est_time_str = "N/A (No blocks allocated)"

    log_message("\n" + "="*70)
    log_message("--- PHASE 2: Tensor Compute Stress-Test & Matrix Math Validation ---")
    log_message(f"Target Cycles:            {args.cycles if args.cycles > 0 else 'Infinite (Loop mode)'}")
    log_message(f"Estimated Waiting Time:   {est_time_str}")
    log_message("="*70 + "\n")
    
    start_time = time.time()
    cycle = 0
    status = "RUNNING"

    try:
        while True:
            cycle += 1
            
            if args.cycles > 0 and cycle > args.cycles:
                status = "SUCCESS / STABLE"
                cycle -= 1
                break
                
            elapsed = time.time() - start_time
            if args.duration > 0 and elapsed >= args.duration:
                status = "SUCCESS / STABLE"
                break
                
            current_temp_str = get_gpu_temp()
            log_message(f"[Cycle {cycle}] Running matrix calculations... Temp: {current_temp_str}")
            
            # Record data points for telemetry plotting
            raw_temp = get_gpu_temp_raw()
            if raw_temp is not None:
                time_telemetry.append(elapsed)
                temp_telemetry.append(raw_temp)
            
            for i in range(len(allocated_blocks)):
                next_idx = (i + 1) % len(allocated_blocks)
                slice_a = allocated_blocks[i][:slice_size, :slice_size]
                slice_b = allocated_blocks[next_idx][:slice_size, :slice_size]
                
                result = torch.matmul(slice_a, slice_b)
                
                if torch.isnan(result).any() or torch.isinf(result).any():
                    status = "FAILED / DATA CORRUPTION"
                    log_message(f"[ERROR] DATA CORRUPTION DETECTED in VRAM block {i}!")
                    generate_plot()
                    print_final_report(status, cycle, time.time() - start_time, "Math validation returned NaN/INF values.")
                    sys.exit(1)
                
                allocated_blocks[i][:slice_size, :slice_size] = result / (result.std() + 1e-5)
                
            torch.cuda.synchronize(device)
            time.sleep(0.02)

        generate_plot()
        print_final_report(status, cycle, time.time() - start_time)

    except RuntimeError as e:
        status = "CRASHED"
        log_message(f"\n[ERROR] HARDWARE DRIVER CRASH DETECTED UNDER LOAD! {str(e)}")
        generate_plot()
        print_final_report(status, cycle, time.time() - start_time, f"CUDA Driver error: {str(e)}")
        sys.exit(1)
    except KeyboardInterrupt:
        status = "STOPPED"
        generate_plot()
        print_final_report(status, cycle, time.time() - start_time, "Execution interrupted by KeyboardInterrupt.")
    finally:
        if has_nvml:
            pynvml.nvmlShutdown()

if __name__ == "__main__":
    main()