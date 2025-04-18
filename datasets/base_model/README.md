# Qwen1.5-0.5B-Chat 基础模型

本目录用于存放Qwen1.5-0.5B-Chat基础模型文件，作为矿机信息处理微调的起点。

## 模型信息

**Qwen1.5-0.5B-Chat**是由阿里云开发的一个轻量级大语言模型，具有以下特点：

- **参数规模**：0.5B（5亿参数）
- **上下文窗口**：2048 tokens
- **支持语言**：中英双语
- **许可证**：商业友好的许可（请参考官方许可条款）
- **优化目标**：在保持基本能力的同时，大幅降低运行资源需求

## 模型下载

模型文件应从Hugging Face官方仓库下载：

```bash
# 方法1：使用huggingface-cli
pip install huggingface_hub
huggingface-cli download Qwen/Qwen1.5-0.5B-Chat --local-dir=./

# 方法2：使用Python API
from huggingface_hub import snapshot_download
snapshot_download(repo_id="Qwen/Qwen1.5-0.5B-Chat", local_dir="./")
```

## 文件结构

下载完成后，目录中应包含以下文件：

```
.
├── config.json                 # 模型配置文件
├── generation_config.json      # 生成参数配置
├── model.safetensors           # 模型权重(safetensors格式)
├── pytorch_model.bin.index.json # 模型索引文件
├── special_tokens_map.json     # 特殊token映射
├── tokenizer.json              # 分词器配置
├── tokenizer_config.json       # 分词器配置
└── tokenizer.model             # 分词器模型
```

## 模型加载

可以使用Hugging Face的Transformers库加载模型：

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

# 加载模型和分词器
model = AutoModelForCausalLM.from_pretrained("./", trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained("./", trust_remote_code=True)

# 确保padding token正确设置
if tokenizer.pad_token_id is None:
    tokenizer.pad_token_id = tokenizer.eos_token_id
```

## 硬件要求

基础模型的最低硬件要求：

- **内存**：至少8GB系统内存
- **GPU**：至少2GB显存（仅在使用GPU时）
- **存储**：约2GB可用磁盘空间

## 与LoRA微调的关系

在矿机信息处理项目中，此基础模型将通过LoRA技术进行微调：

1. 基础模型的参数保持不变
2. 微调时仅训练少量LoRA适配器参数
3. 推理时将基础模型与LoRA适配器结合，或使用合并后的模型

## 注意事项

1. 确保下载完整的模型文件，不完整的模型会导致加载失败
2. 如果模型文件有更新，请重新下载最新版本
3. 请遵循Qwen模型的使用许可和条款
4. 基础模型的性能和能力会直接影响微调后的模型，请确保使用正确的版本

## 参考链接

- [Qwen官方GitHub仓库](https://github.com/QwenLM/Qwen)
- [Qwen1.5-0.5B-Chat Hugging Face页面](https://huggingface.co/Qwen/Qwen1.5-0.5B-Chat)
- [Qwen技术报告](https://arxiv.org/abs/2309.16609)
