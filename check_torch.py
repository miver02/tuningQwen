import torch

print(f"PyTorch版本: {torch.__version__}")
print(f"CUDA是否可用: {torch.cuda.is_available()}")

try:
    if hasattr(torch.version, 'cuda'):
        print(f"PyTorch编译的CUDA版本: {torch.version.cuda}")
    else:
        print("PyTorch编译的CUDA版本: 不适用（CPU版本）")
except Exception as e:
    print(f"获取CUDA版本时发生错误: {e}")

try:
    if torch.cuda.is_available():
        print(f"GPU数量: {torch.cuda.device_count()}")
        print(f"当前GPU: {torch.cuda.current_device()}")
        print(f"GPU名称: {torch.cuda.get_device_name(0)}")
    else:
        print("\nCUDA不可用的可能原因:")
        print("1. 您安装的是PyTorch CPU版本")
        print("2. NVIDIA驱动程序与PyTorch支持的CUDA版本不兼容")
        print("3. 系统环境变量PATH中缺少CUDA库路径")
except Exception as e:
    print(f"获取GPU信息时发生错误: {e}") 