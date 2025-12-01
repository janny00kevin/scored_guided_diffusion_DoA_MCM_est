import torch
import time

def check_gpu():
    print("-" * 30)
    print(f"PyTorch verson: {torch.__version__}")
    
    # 1. ?? CUDA ????
    if torch.cuda.is_available():
        print(" CUDA (GPU) is available")
        
        # ?? GPU ?????
        gpu_count = torch.cuda.device_count()
        print(f"GPU num: {gpu_count}")
        for i in range(gpu_count):
            print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
            
        device = torch.device("cuda")
    else:
        print("? CUDA not accessible,turning to CPU?")
        device = torch.device("cpu")
        
    print("-" * 30)
    return device

def simple_benchmark(device):
    print(f"running simple matrix calculation on {device}...")
    
    try:
        # 2. ?????? (Tensor) ??? GPU
        # ???? 4096 x 4096 ??? (????,??? GPU ????)
        size = 4096
        x = torch.rand(size, size).to(device)
        y = torch.rand(size, size).to(device)
        
        torch.cuda.synchronize() if device.type == 'cuda' else None
        start_time = time.time()
        
        # 3. ??????
        result = torch.matmul(x, y)
        
        torch.cuda.synchronize() if device.type == 'cuda' else None
        end_time = time.time()
        
        print("complete")
        print(f"runtime: {end_time - start_time:.4f} sec")
        print(f"tensor  are on {result.device}")
        
    except Exception as e:
        print(f"error: {e}")
        
def gpu_stress_test(duration_minutes=10, matrix_size=10000):
    """
    Runs a heavy GPU load for a specified duration using matrix multiplication.
    """
    print("=" * 50)
    print("PyTorch GPU Stress Test")
    print("=" * 50)

    # 1. Check for CUDA
    if not torch.cuda.is_available():
        print("[ERROR] CUDA is not available. This script requires a GPU.")
        sys.exit(1)

    # Get device info
    device = torch.device("cuda")
    gpu_name = torch.cuda.get_device_name(0)
    print(f"[INFO] Detected GPU: {gpu_name}")
    print(f"[INFO] Target Duration: {duration_minutes} minutes")
    print(f"[INFO] Matrix Size: {matrix_size}x{matrix_size}")
    
    # 2. Prepare Data
    print("[INFO] Allocating tensors on GPU...")
    try:
        # Create two large random matrices on the GPU
        # Float32 takes 4 bytes. 10000x10000x4 bytes ~= 400MB per tensor.
        # We need A, B, and the result C, so approx 1.2GB VRAM usage.
        tensor_a = torch.rand(matrix_size, matrix_size, device=device)
        tensor_b = torch.rand(matrix_size, matrix_size, device=device)
    except RuntimeError as e:
        print(f"[ERROR] OOM or Allocation failed: {e}")
        return

    # 3. Start Stress Loop
    start_time = time.time()
    end_time = start_time + (duration_minutes * 60)
    
    print("[INFO] Starting stress test... (Press Ctrl+C to stop manually)")
    
    iteration = 0
    
    try:
        while time.time() < end_time:
            # Perform heavy matrix multiplication
            # This is compute-bound and good for heating up the GPU
            _ = torch.matmul(tensor_a, tensor_b)
            
            # Synchronization is crucial! 
            # Without this, Python sends commands faster than GPU processes,
            # potentially causing OOM or inaccurate timing.
            torch.cuda.synchronize()

            iteration += 1
            
            # Print status every 10 iterations to keep the log clean
            if iteration % 10 == 0:
                elapsed = time.time() - start_time
                remaining = (duration_minutes * 60) - elapsed
                print(f"[RUNNING] Elapsed: {elapsed/60:.1f} min | Remaining: {remaining/60:.1f} min | Iterations: {iteration}")

    except KeyboardInterrupt:
        print("\n[INFO] Test interrupted by user.")
    
    # 4. Finish
    total_time = time.time() - start_time
    print("=" * 50)
    print(f"[DONE] Test finished.")
    print(f"[INFO] Total Runtime: {total_time/60:.2f} minutes")
    print("=" * 50)

if __name__ == "__main__":
    device = check_gpu()
    simple_benchmark(device)
    gpu_stress_test(duration_minutes=10, matrix_size=12000)