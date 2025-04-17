# Qwen1.5-0.5B-Chat 矿机信息处理模型训练

这个项目用于训练 Qwen1.5-0.5B-Chat 模型，使其能够识别矿机信息并返回标准化的矿机名称和规格。

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
├── main.py                # 主要训练代码
├── test_model.py          # 测试训练好的模型
├── requirements.txt       # 依赖项
└── README.md              # 说明文档
```

## 环境配置

1. 安装依赖项

```bash
pip install -r requirements.txt
```

2. 确保基础模型已经下载并放在 `datasets/base_model` 目录中

如果没有预先下载模型，可以使用以下方法：

```bash
# 创建模型目录
mkdir -p datasets/base_model

# 使用 huggingface-cli 下载模型
pip install huggingface_hub
huggingface-cli download Qwen/Qwen1.5-0.5B-Chat --local-dir datasets/base_model
```

## 训练模型

运行以下命令开始训练：

```bash
python main.py
```

训练参数已在代码中配置好，适用于 Qwen1.5-0.5B-Chat 模型：
- 使用 LoRA 进行微调
- 学习率: 1e-4
- 批处理大小: 8
- 训练轮次: 5
- 最大序列长度: 512
- 权重衰减: 0.01
- 混合精度训练: 启用
- 梯度检查点: 启用
- 输出目录: datasets/final_model

### 训练后的模型保存

训练完成后，以下内容会自动保存：

1. **LoRA适配器模型**: 保存在 `datasets/final_model` 和 `datasets/final_model/lora_model` 目录下
2. **合并后的完整模型**: 保存在 `datasets/final_model/merged_model` 目录下
3. **分词器**: 保存在主目录和各子目录下

即使某些保存步骤失败，代码也会尝试保存其他部分，并提供详细的错误信息。

### 修改训练参数

如果需要修改训练参数，可以直接修改 `main.py` 中的 `TrainingArguments` 部分。

### 故障排除

如果遇到以下问题，可以尝试相应的解决方案：

1. 内存不足
   - 减少批处理大小 (修改 `per_device_train_batch_size`)
   - 增加梯度累积步骤 (修改 `gradient_accumulation_steps`)
   - 确保开启梯度检查点 (`gradient_checkpointing=True`)

2. 训练出错
   - 查看详细的错误信息，代码已添加额外的错误处理和日志
   - 检查数据格式是否正确
   - 确保模型路径存在且包含完整的模型文件

3. 模型加载失败
   - 确保已正确安装所有依赖项
   - 确保指定的模型路径是正确的
   - 尝试重新下载模型

4. DeepSpeed相关错误
   - 代码已增加错误处理，可以在不支持DeepSpeed的环境中正常运行
   - 如遇到特定的DeepSpeed错误，请检查错误日志并据此调整参数

## 测试模型

训练完成后，可以使用以下命令测试模型效果：

```bash
python test_model.py
```

默认参数:
- 模型路径: datasets/final_model
- 测试数据: datasets/train_data/train_v1.jsonl
- 测试样本数: 5
- 结果输出: test_results.jsonl

### 高级测试选项

测试脚本支持多种模型加载方式，可以根据需要选择：

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

测试脚本会自动检测模型的类型（LoRA、合并模型或普通模型），并适当地加载。它将输出每个样本的结果和总体准确率，并会生成一个包含详细结果的JSON文件。

## 使用模型进行推理

可以根据以下示例代码使用训练好的模型进行推理：

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

### 使用LoRA模型推理

如果您想使用LoRA模型（而不是合并后的模型）进行推理，可以使用以下代码：

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch

# 加载基础模型
base_model_path = "datasets/base_model"
base_model = AutoModelForCausalLM.from_pretrained(
    base_model_path,
    trust_remote_code=True
)

# 加载LoRA权重
lora_path = "datasets/final_model/lora_model"
model = PeftModel.from_pretrained(base_model, lora_path)

# 加载分词器
tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)

# 确保padding token正确设置
if tokenizer.pad_token_id is None:
    tokenizer.pad_token_id = tokenizer.eos_token_id

# 进行推理的其余部分与上面相同
```

## 数据格式

训练数据格式为JSONL，每行包含一个JSON对象，格式如下：

```json
{"input": "m50s 26w 124t gtd 1020u $1020.0", "output": "MicroBT Whatsminer M50S - 124.0th/t $1020.0"}
```

- `input`: 原始矿机信息字符串
- `output`: 标准化的矿机名称和规格

## 进阶调整

### 自定义LoRA参数

如果需要调整LoRA参数，可以修改 `main.py` 中的 `LoraArguments` 类：

```python
@dataclass
class LoraArguments:
    lora_r: int = 8  # LoRA的秩
    lora_alpha: int = 16  # LoRA的放大因子
    lora_dropout: float = 0.05  # LoRA的dropout率
    # 目标模块，这些是Qwen1.5模型中适合应用LoRA的层
    lora_target_modules: List[str] = field(
        default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    )
    lora_bias: str = "none"  # 偏置项处理方式
```

### 性能优化

如果在低资源环境中运行，可以考虑：

1. 减小模型序列长度 (`model_max_length`)
2. 使用CPU训练 (移除fp16设置)
3. 增加梯度累积步骤以使用更小的批处理大小

## 注意事项

1. 训练时如果遇到内存不足，可以减小批处理大小或使用梯度累积
2. 对于较小的模型（如0.5B参数量），训练时间相对较短
3. 如果模型预测效果不佳，可以尝试增加训练轮次或调整学习率
4. 代码中包含详细的错误处理和日志记录，有助于诊断问题
5. 如果模型合并失败，您仍然可以使用保存的LoRA权重进行推理