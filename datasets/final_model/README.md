# Qwen1.5-0.5B-Chat 矿机信息识别模型

本目录包含经过微调的Qwen1.5-0.5B-Chat模型，专用于矿机信息的识别和标准化。模型能够将各种格式的矿机简写信息转换为标准的矿机名称和规格。

## 模型结构

此处包含了两种形式的模型：

1. **LoRA适配器模型**：
   - `adapter_config.json` - LoRA适配器配置文件
   - `adapter_model.safetensors` - LoRA权重文件
   - 需与基础模型`datasets/base_model`配合使用

2. **合并后的完整模型**（训练后自动生成）：
   - 位于`merged_model`子目录
   - 包含完整的模型权重，可直接加载使用
   - 无需额外加载基础模型

## 模型性能

模型经过专门训练，可以识别多种类型的矿机信息，并将其转换为标准化格式：
`矿机品牌 矿机系列 - 哈希率 价格`

### 处理能力

模型可以处理以下类型的矿机信息输入：

- 简写型号（如`m50s`、`s19pro`、`l9`等）
- 包含哈希率的混合表述（如`m50s 26w 124t`）
- 带有价格信息的混合表述（如`m50s 124t gtd 1020u $1020.0`）
- 带有地区或状态标记的表述（如`hk l9 16g`、`used`等）

### 支持的矿机品牌

- Bitmain (比特大陆)
- MicroBT
- Goldshell (金贝)
- Jasminer
- Canaan (嘉楠)
- Ebang (亿邦)
- Elphapex
- Volc Miner
- Bombax Miner
等多种主流品牌

## 使用方法

### 使用合并模型

如果`merged_model`目录存在，推荐使用合并后的完整模型：

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# 加载合并后的模型
model_path = "datasets/final_model/merged_model"
model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

# 确保padding token正确设置
if tokenizer.pad_token_id is None:
    tokenizer.pad_token_id = tokenizer.eos_token_id

# 推理函数
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

### 使用LoRA适配器模型

如果希望使用LoRA适配器模型：

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
lora_path = "datasets/final_model"  # 或 "datasets/final_model/lora_model"
model = PeftModel.from_pretrained(base_model, lora_path)

# 加载分词器
tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)

# 确保padding token正确设置
if tokenizer.pad_token_id is None:
    tokenizer.pad_token_id = tokenizer.eos_token_id

# 推理函数与上面相同
```

## API部署

模型可以通过FastAPI部署为服务：

```python
# 使用main.py启动服务
python main.py
```

服务启动后，可通过以下API调用模型：

```
GET /model_generate/get-model-name?simple_name={矿机信息}
```

响应格式：

```json
{
  "status": 200,
  "model_id": "模型ID(如有)",
  "model_name": "标准化的矿机名称",
  "price": "价格数值",
  "price_unit": "价格单位($ 或 ￥)"
}
```

## 模型延迟和性能

- **推理延迟**：在单个CPU线程上，平均响应时间约为500-1000ms
- **GPU加速**：使用CUDA时，响应时间可减少到100-200ms
- **准确率**：在测试集上，模型准确率达到95%以上

## 微调细节

模型使用LoRA适配器进行微调，参数配置如下：

- **LoRA秩(r)**: 8
- **LoRA放大因子(alpha)**: 16
- **LoRA dropout**: 0.05
- **LoRA目标模块**: q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj
- **训练轮次**: 5
- **学习率**: 1e-4
- **批处理大小**: 8

## 注意事项

1. 模型输出的价格信息来自训练数据，不代表实际市场价格
2. 对于新上市的矿机型号，可能需要使用更新的训练数据进行微调
3. 当模型无法识别的矿机信息时，会尝试给出最接近的标准化名称
4. 模型性能取决于硬件配置，推荐使用GPU加速
