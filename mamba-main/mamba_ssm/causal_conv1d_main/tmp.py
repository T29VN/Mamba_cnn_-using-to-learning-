import torch

print("✅ PyTorch đã cài:", torch.__version__)
print("🔧 CUDA phiên bản đi kèm:", torch.version.cuda)
print("🚀 GPU khả dụng:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("📦 Tên GPU:", torch.cuda.get_device_name(0))
else:
    print("⚠️ Không tìm thấy GPU CUDA.")
