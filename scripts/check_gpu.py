#!/usr/bin/env python
"""
Script to check GPU availability and specifications.
"""
import argparse

import torch


def print_detailed_gpu_info():
    """Print detailed information about available GPUs."""
    print("\n===== GPU Information =====")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"CUDNN version: {torch.backends.cudnn.version()}")
        print(f"Number of GPUs: {torch.cuda.device_count()}")
        
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            print(f"\nGPU {i}: {props.name}")
            print(f"  Compute capability: {props.major}.{props.minor}")
            print(f"  Total memory: {props.total_memory / 1e9:.2f} GB")
            print(f"  CUDA cores: {props.multi_processor_count}")
            if hasattr(props, 'max_memory_allocated'):
                print(f"  Memory allocated: {torch.cuda.max_memory_allocated(i) / 1e9:.2f} GB")
                print(f"  Memory reserved: {torch.cuda.memory_reserved(i) / 1e9:.2f} GB")
            print(f"  Memory clock rate: {props.memory_clock_rate / 1e3} MHz")
            print(f"  Memory bus width: {props.memory_bus_width} bits")
            
        # Additional information for debugging
        print("\n===== CUDA Environment =====")
        print(f"Current device: {torch.cuda.current_device()}")
        print(f"Default device: cuda:{torch.cuda.current_device()}")
        print(f"Tensor inside GPU: {torch.tensor([1.0]).cuda().device}")
    else:
        print("\nNo CUDA-compatible GPU found. This system is using CPU only.")
        
        # Check for MPS (Apple Silicon GPU) if on Mac
        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            print("\n===== MPS (Apple Silicon GPU) Information =====")
            print(f"MPS available: {torch.backends.mps.is_available()}")
            print(f"MPS built: {torch.backends.mps.is_built()}")
            try:
                # Create a small tensor on MPS device
                test_tensor = torch.tensor([1.0], device="mps")
                print(f"MPS device supported and working: Yes")
                print(f"MPS test tensor: {test_tensor.device}")
            except Exception as e:
                print(f"MPS device error: {e}")


def main():
    parser = argparse.ArgumentParser(description="Check GPU availability and specifications.")
    parser.add_argument("--test", action="store_true", help="Run a small GPU performance test")
    args = parser.parse_args()
    
    print_detailed_gpu_info()
    
    # Optionally run a small performance test
    if args.test and torch.cuda.is_available():
        print("\n===== Performance Test =====")
        print("Running a simple matrix multiplication test...")
        
        # Create random tensors
        size = 5000
        a = torch.randn(size, size, device="cuda")
        b = torch.randn(size, size, device="cuda")
        
        # Warmup
        torch.cuda.synchronize()
        for _ in range(5):
            c = torch.matmul(a, b)
        torch.cuda.synchronize()
        
        # Timing
        import time
        start = time.time()
        for _ in range(10):
            c = torch.matmul(a, b)
        torch.cuda.synchronize()
        end = time.time()
        
        print(f"Time for 10 iterations of {size}x{size} matrix multiplication: {(end - start):.4f} seconds")
        print(f"Average per iteration: {(end - start) / 10:.4f} seconds")
    
    print("\nGPU check complete!")


if __name__ == "__main__":
    main()