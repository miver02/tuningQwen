# Qwen1.5-0.5B-Chat 矿机信息处理模型训练

## 项目概述

本项目旨在训练 Qwen1.5-0.5B-Chat 模型，使其能够识别各种格式的矿机信息，并将其转换为标准化的矿机名称和规格格式。在加密货币挖矿行业中，矿机信息经常以不同的简写和格式出现，本项目通过微调大语言模型，实现对这些非标准信息的智能解析。

### 主要功能

- 识别简写、非标准的矿机信息文本
- 将信息转换为标准格式：`矿机品牌 矿机系列 - 哈希率 价格`
- 提供模型训练、测试和部署的完整流程
- 支持通过API调用进行实时推理

### 应用场景

- 加密货币交易平台
- 矿机交易市场
- 矿机性能分析工具
- 挖矿收益计算器

## 项目结构

```
.
├── datasets/
│   ├── base_model/        # Qwen1.5-0.5B-Chat 基础模型
│   ├── final_model/       # 训练后的模型保存位置
│       ├── merged_model/  # 合并后的完整模型（训练后自动生成）
│       ├── lora_model/    # 保存的LoRA权重（训练后自动生成）
│       └── tokenizer/     # 分词器（训练后自动生成）
│   └── train_data/        # 训练数据
│       └── train_v1.jsonl # 训练数据文件
├── main.py                # FastAPI服务器实现，用于模型推理
├── train.py               # 主要训练代码，包含完整的模型训练逻辑
├── test_model.py          # 测试训练好的模型性能
├── handle_str.py          # 字符串处理相关功能
├── check_torch.py         # 检查PyTorch环境
├── install_pytorch.py     # 安装PyTorch相关依赖
├── start.sh               # 启动服务的脚本
├── Dockerfile             # Docker容器定义文件
├── requirements.txt       # 项目依赖项
└── README.md              # 项目说明文档
```

## 技术细节

### 基础模型

本项目使用 [Qwen1.5-0.5B-Chat](https://huggingface.co/Qwen/Qwen1.5-0.5B-Chat) 作为基础模型。这是一个由阿里云开发的轻量级大语言模型，具有以下特点：

- 参数量：0.5B（5亿参数）
- 上下文窗口：2048 tokens
- 支持中英双语对话
- 优化的推理速度和资源消耗

### 微调方法

项目采用 [LoRA (Low-Rank Adaptation)](https://arxiv.org/abs/2106.09685) 技术进行参数高效微调：

- **LoRA秩(r)**: 8
- **LoRA放大因子(alpha)**: 16
- **LoRA丢弃率(dropout)**: 0.05
- **目标模块**: q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj
- **偏置处理**: none

LoRA通过训练低秩适配矩阵来调整预训练权重，大大减少了可训练参数的数量，降低了计算资源需求。

### 训练配置

- **最大序列长度**: 512 tokens
- **批处理大小**: 8
- **梯度累积步骤**: 2
- **学习率**: 1e-4
- **权重衰减**: 0.01
- **预热比例**: 0.03
- **学习率调度器**: cosine
- **训练轮次**: 5
- **优化器**: adamw_torch
- **混合精度训练**: fp16
- **梯度检查点**: 启用

## 环境配置

### 硬件要求

- **最低要求**: 4GB GPU内存或8GB系统内存
- **推荐配置**: 8GB+ GPU内存，16GB系统内存

### 软件依赖

1. 安装Python依赖项

```bash
pip install -r requirements.txt
```

2. 检查GPU和PyTorch环境（可选）

```bash
python check_torch.py
```

3. 如需自动安装适合的PyTorch版本（可选）

```bash
python install_pytorch.py
```

### 获取基础模型

确保基础模型已经下载并放在 `datasets/base_model` 目录中。可以通过以下方法下载：

```bash
# 创建模型目录
mkdir -p datasets/base_model

# 使用 huggingface-cli 下载模型
pip install huggingface_hub
huggingface-cli download Qwen/Qwen1.5-0.5B-Chat --local-dir datasets/base_model
```

或者使用Python代码下载：

```python
from huggingface_hub import snapshot_download

snapshot_download(repo_id="Qwen/Qwen1.5-0.5B-Chat", local_dir="datasets/base_model")
```

## 训练模型

### 开始训练

运行以下命令开始训练：

```bash
python train.py
```

默认配置已经针对Qwen1.5-0.5B-Chat模型和矿机信息处理任务进行了优化。

### 自定义训练参数

如需修改训练参数，可以编辑`train.py`中的`TrainingArguments`类：

```python
@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(default=512)
    use_lora: bool = True
    output_dir: str = field(default="datasets/final_model")
    num_train_epochs: float = field(default=5.0)
    per_device_train_batch_size: int = field(default=8)
    gradient_accumulation_steps: int = field(default=2)
    learning_rate: float = field(default=1e-4)
    weight_decay: float = field(default=0.01)
    warmup_ratio: float = field(default=0.03)
    lr_scheduler_type: str = field(default="cosine")
    logging_steps: int = field(default=10)
    save_strategy: str = field(default="epoch")
    save_total_limit: int = field(default=3)
    fp16: bool = field(default=True)
    bf16: bool = field(default=False)
    gradient_checkpointing: bool = field(default=True)
```

### 训练后的模型保存

训练完成后，以下内容会自动保存：

1. **LoRA适配器模型**: 保存在 `datasets/final_model` 和 `datasets/final_model/lora_model` 目录下
2. **合并后的完整模型**: 保存在 `datasets/final_model/merged_model` 目录下
3. **分词器**: 保存在主目录和各子目录下

即使某些保存步骤失败，代码也会尝试保存其他部分，并提供详细的错误信息。

### 训练数据格式

训练数据格式为JSONL，每行包含一个JSON对象，格式如下：

```json
{"input": "m50s 26w 124t gtd 1020u $1020.0", "output": "MicroBT Whatsminer M50S - 124.0th/t $1020.0"}
```

- `input`: 原始矿机信息字符串，通常包含简写、非标准格式
- `output`: 标准化的矿机名称和规格，遵循`矿机品牌 矿机系列 - 哈希率 价格`格式

## 测试模型

训练完成后，可以使用以下命令测试模型效果：

```bash
python test_model.py
```

### 测试参数

测试脚本支持多种参数配置：

```bash
# 使用合并后的完整模型测试（默认行为）
python test_model.py

# 使用LoRA模型测试（需指定基础模型）
python test_model.py --use_lora --base_model_path datasets/base_model

# 使用CPU运行测试
python test_model.py --use_cpu

# 指定其他参数
python test_model.py --model_path [模型路径] --test_file [测试文件] --num_samples [样本数量] --output_file [输出文件]
```

### 测试输出

测试脚本会生成以下输出：

1. 控制台输出：显示每个样本的输入、预期输出和模型实际输出，以及总体准确率
2. 结果文件：默认为`test_results.jsonl`，包含详细的测试结果数据

## 部署和使用

### 启动推理服务

项目提供了基于FastAPI的推理服务，可通过以下命令启动：

```bash
python main.py
```

或使用提供的脚本：

```bash
bash start.sh
```

服务默认在`http://0.0.0.0:8000`上运行，提供以下API端点：

- `/model_generate/get-model-name?simple_name={矿机信息}`：将简略矿机信息转换为标准格式

### 使用Docker部署

项目包含Dockerfile，可以构建Docker镜像进行部署：

```bash
# 构建Docker镜像
docker build -t mining-model .

# 运行Docker容器
docker run -p 8000:8000 mining-model
```

### 代码中直接使用

可以根据以下示例代码在Python程序中使用训练好的模型：

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# 加载模型和分词器 - 优先使用合并后的模型
model_path = "datasets/final_model/merged_model"  # 使用合并后的模型
model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

# 确保padding token正确设置
if tokenizer.pad_token_id is None:
    tokenizer.pad_token_id = tokenizer.eos_token_id

# 创建一个函数用于生成回复
def process_mining_info(input_text):
    # 构建Qwen1.5-Chat格式的对话
    conversation = f"<|im_start|>user\n我将为你提供一个矿机信息，你需要返回标准格式的矿机名称和规格，格式为：矿机品牌 矿机系列 - 哈希率 价格。\n\n矿机信息：{input_text}<|im_end|>\n<|im_start|>assistant\n"
    
    # 编码输入
    inputs = tokenizer(conversation, return_tensors="pt").to(model.device)
    
    # 生成输出
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=50,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.convert_tokens_to_ids("<|im_end|>")
        )
    
    # 解码输出
    output_text = tokenizer.decode(outputs[0], skip_special_tokens=False)
    
    # 提取助手回复部分
    assistant_part = output_text.split("<|im_start|>assistant\n")[1]
    if "<|im_end|>" in assistant_part:
        return assistant_part.split("<|im_end|>")[0].strip()
    else:
        return assistant_part.strip()

# 示例使用
sample_input = "m50s 26w 124t gtd 1020u $1020.0"
result = process_mining_info(sample_input)
print(f"输入: {sample_input}")
print(f"输出: {result}")
```

## 故障排除

### 内存不足

如果遇到内存不足问题，可尝试：

1. 减少批处理大小（修改 `per_device_train_batch_size`）
2. 增加梯度累积步骤（修改 `gradient_accumulation_steps`）
3. 确保开启梯度检查点（`gradient_checkpointing=True`）
4. 使用CPU进行训练（删除或注释掉fp16相关设置）

### 模型加载失败

如果模型加载失败，请检查：

1. 模型路径是否正确
2. 是否已完全下载模型文件
3. 依赖包版本是否兼容
4. GPU内存是否足够

可以尝试使用以下代码检查模型目录是否完整：

```python
import os
model_path = "datasets/base_model"
print(f"模型路径中的文件: {os.listdir(model_path)}")
```

### 训练过程中断

如果训练过程中断，可以：

1. 检查日志了解具体错误
2. 确保磁盘空间充足
3. 尝试使用更小的学习率重新开始训练
4. 如果是GPU内存问题，调小批处理大小

### 训练结果不理想

如果训练后的模型效果不佳，可以尝试：

1. 增加训练轮次
2. 调整学习率
3. 增加训练数据量和多样性
4. 修改LoRA参数（如增加r值）

## 进阶调整

### 自定义LoRA参数

可以修改 `train.py` 中的 `LoraArguments` 类来调整LoRA配置：

```python
@dataclass
class LoraArguments:
    lora_r: int = 8                # LoRA的秩
    lora_alpha: int = 16           # LoRA的放大因子
    lora_dropout: float = 0.05     # LoRA的dropout率
    lora_target_modules: List[str] = field(
        default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    )
    lora_bias: str = "none"        # 偏置项处理方式
```

### 性能优化

在低资源环境中，可以考虑以下优化：

1. 使用更小的序列长度（降低 `model_max_length`）
2. 禁用fp16混合精度训练（设置 `fp16=False`）
3. 增加梯度累积步骤，减小批处理大小
4. 使用CPU进行训练（确保移除GPU相关设置）

### 自定义推理参数

推理时可以调整生成参数以平衡质量和速度：

```python
outputs = model.generate(
    **inputs,
    max_new_tokens=50,        # 可以调小以加快速度
    do_sample=True,           # 启用采样以增加多样性
    temperature=0.7,          # 控制随机性 (0.2-1.0)
    top_p=0.9,                # 控制采样范围
    repetition_penalty=1.1,   # 避免重复内容
    pad_token_id=tokenizer.pad_token_id,
    eos_token_id=tokenizer.convert_tokens_to_ids("<|im_end|>")
)
```

## 贡献指南

欢迎对本项目做出贡献：

1. Fork 本仓库
2. 创建你的特性分支 (`git checkout -b feature/amazing-feature`)
3. 提交你的更改 (`git commit -m 'Add some amazing feature'`)
4. 推送到分支 (`git push origin feature/amazing-feature`)
5. 开启一个 Pull Request

## 注意事项

1. 训练时如果遇到内存不足，可以减小批处理大小或使用梯度累积
2. 对于较小的模型（如0.5B参数量），训练时间相对较短
3. 如果模型预测效果不佳，可以尝试增加训练轮次或调整学习率
4. 代码中包含详细的错误处理和日志记录，有助于诊断问题
5. 如果模型合并失败，您仍然可以使用保存的LoRA权重进行推理

## 许可证

[选择合适的许可证]

## 致谢

- 感谢阿里云提供的Qwen模型
- 感谢Hugging Face提供的Transformers库
- 感谢PEFT库提供的LoRA实现