# run file to test if torch, gpu, onnx is working.
import torch

print("PyTorch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("GPU name:", torch.cuda.get_device_name(0))
    print("Device count:", torch.cuda.device_count())

# Simple tensor test
x = torch.rand(3, 3)
y = torch.rand(3, 3)
z = torch.mm(x, y)

print("Matrix multiplication result:\n", z)

# Optional: move to GPU if available
if torch.cuda.is_available():
    x_gpu = x.to("cuda")
    y_gpu = y.to("cuda")
    z_gpu = torch.mm(x_gpu, y_gpu)
    print("GPU matrix multiplication successful:", z_gpu.is_cuda)
