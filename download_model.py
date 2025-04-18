#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
简单的脚本用于下载Qwen1.5-0.5B-Chat模型
"""

import os
import sys
from huggingface_hub import snapshot_download
from tqdm.auto import tqdm

def download_model():
    """下载Qwen1.5-0.5B-Chat模型到datasets/base_model目录"""
    
    # 目标目录
    output_dir = "datasets/base_model"
    
    # 创建目录（如果不存在）
    os.makedirs(output_dir, exist_ok=True)
    print(f"下载目录: {output_dir}")
    
    try:
        # 使用snapshot_download下载模型
        print("开始下载Qwen1.5-0.5B-Chat模型...")
        snapshot_download(
            repo_id="Qwen/Qwen1.5-0.5B-Chat",
            local_dir=output_dir,
            local_dir_use_symlinks=False
        )
        print(f"模型下载完成！保存在: {output_dir}")
        
        # 简单验证
        required_files = ["config.json", "tokenizer.model", "model.safetensors"]
        missing = [f for f in required_files if not os.path.exists(os.path.join(output_dir, f))]
        
        if missing:
            print(f"警告：部分文件可能未下载完全: {', '.join(missing)}")
        else:
            print("模型文件验证通过!")
            
    except Exception as e:
        print(f"下载出错: {e}")
        print("\n尝试使用以下命令手动下载:")
        print("pip install huggingface_hub")
        print(f"huggingface-cli download Qwen/Qwen1.5-0.5B-Chat --local-dir {output_dir}")
        
if __name__ == "__main__":
    # 检查是否安装了huggingface_hub
    try:
        import huggingface_hub
    except ImportError:
        print("缺少huggingface_hub库，正在安装...")
        os.system(f"{sys.executable} -m pip install huggingface_hub")
        print("安装完成")
        
    # 检查是否安装了tqdm
    try:
        import tqdm
    except ImportError:
        print("缺少tqdm库，正在安装...")
        os.system(f"{sys.executable} -m pip install tqdm")
        print("安装完成")
        
    # 下载模型
    download_model()
