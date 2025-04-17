import subprocess
import sys

def install_pytorch():
    print("开始安装支持CUDA的PyTorch...")
    # 卸载现有PyTorch
    subprocess.check_call([sys.executable, "-m", "pip", "uninstall", "-y", "torch", "torchvision", "torchaudio"])
    
    # 安装支持CUDA的PyTorch (使用国内镜像)
    try:
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", 
            "torch", "torchvision", "torchaudio", 
            "--index-url", "https://download.pytorch.org/whl/cu118"
        ])
        print("PyTorch安装成功完成！")
    except Exception as e:
        print(f"安装出错: {str(e)}")
        print("尝试使用备用方法...")
        try:
            subprocess.check_call([
                sys.executable, "-m", "pip", "install", 
                "torch==2.2.0", "torchvision", "torchaudio", 
                "--index-url", "https://download.pytorch.org/whl/cu118"
            ])
            print("PyTorch安装成功完成！")
        except Exception as e:
            print(f"安装再次出错: {str(e)}")
            print("请尝试手动安装。")

if __name__ == "__main__":
    install_pytorch() 