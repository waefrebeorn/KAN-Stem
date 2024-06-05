import os
import psutil
import torch
import time  # For timing the initialization
import warnings

warnings.filterwarnings("ignore", message="Lazy modules are a new feature under heavy development")
warnings.filterwarnings("ignore", message="oneDNN custom operations are on. You may see slightly different numerical results due to floating-point round-off errors from different computation orders.")

# Function to print memory usage
def print_memory_usage(step):
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    print(f"{step}: Memory usage: {mem_info.rss / 1024**2:.2f} MB")

# Print initial memory usage
print_memory_usage("Before PyTorch import")

# Import PyTorch and time the import
start_time = time.time()
import torch
end_time = time.time()

print(f"PyTorch imported in {end_time - start_time:.2f} seconds")

# Check for CUDA availability
print("\nChecking CUDA availability...")
if torch.cuda.is_available():
    print("CUDA is available!")
    num_devices = torch.cuda.device_count()
    print(f"Found {num_devices} CUDA device(s):")
    for i in range(num_devices):
        print(f"- Device {i}: {torch.cuda.get_device_name(i)}")
else:
    print("CUDA is not available. Using CPU.")

# Create a CUDA tensor if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
x = torch.randn(1000000).to(device)

# Perform a simple computation to verify CUDA functionality
if torch.cuda.is_available():
    print("\nPerforming a test computation on CUDA...")
    start_time = time.time()
    y = torch.matmul(x, x.T)
    end_time = time.time()
    print(f"Computation completed in {end_time - start_time:.2f} seconds")

# Print final memory usage
print_memory_usage("After CUDA test")
