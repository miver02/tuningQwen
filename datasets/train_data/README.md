# 矿机信息训练数据

本目录包含用于训练Qwen1.5-0.5B-Chat模型的矿机信息数据集。这些数据经过精心收集和整理，用于教会模型识别各种格式的矿机信息，并将其标准化。

## 数据格式

训练数据以JSONL格式存储，每行是一个独立的JSON对象，包含两个字段：

- `input`: 原始矿机信息字符串，通常是非标准化的简写形式
- `output`: 标准化后的矿机名称和规格，遵循`矿机品牌 矿机系列 - 哈希率 价格`格式

示例：
```json
{"input": "m50s 26w 124t gtd 1020u $1020.0", "output": "MicroBT Whatsminer M50S - 124.0th/t $1020.0"}
```

## 数据内容说明

数据集包含多种类型的矿机信息，涵盖了市场上主流的矿机品牌和型号，包括但不限于：

1. **比特大陆(Bitmain)** 系列矿机：
   - Antminer S19, S19 Pro, S19 XP, S19K Pro
   - Antminer T21
   - Antminer L7, L9
   - Antminer K7, KS5, KS5 Pro, KS7
   - Antminer E9 Pro

2. **MicroBT** 系列矿机：
   - Whatsminer M50, M50S, M50S++
   - Whatsminer M60, M60S, M60S+
   - Whatsminer M61

3. **金贝(Goldshell)** 系列矿机：
   - Mini Doge, Mini Doge+, Mini Doge III+
   - DG Home, DG Home1
   - AL3, AL Max
   - AE Box, AE Box Pro

4. **其他品牌**矿机：
   - Jasminer X16-Q
   - Canaan Avalon Miner A1466, A1566
   - Bombax Miner EZ100-C
   - Ebang Ebit E11
   - Elphapex DG1+
   - Volc Miner D1, D1 Mini

## 数据命名规则说明

数据中的矿机信息包含多种信息编码方式，以下是一些常见的格式和缩写说明：

1. **哈希率单位**：
   - `t` 或 `th/t`: TH/s (太哈希/秒)
   - `g` 或 `gh/t`: GH/s (千兆哈希/秒)
   - `m` 或 `mh/t`: MH/s (兆哈希/秒)

2. **功率标识**：
   - `w`: 表示瓦特，如 `26w` 表示26瓦特

3. **价格表示**：
   - `$数字`: 美元价格
   - `￥数字`: 人民币价格
   - `数字u`: 美元价格，如 `1020u` 表示1020美元

4. **其他常见缩写**：
   - `gtd`: guaranteed (保证)
   - `hyd`: hydro (水冷版)
   - `hk`, `sz`: 地区代码，如香港(hk)、深圳(sz)
   - `mix`: 混合型号
   - `rb`: 翻新(refurbished)

## 数据处理和标准化

原始数据经过以下处理步骤：

1. **数据收集**：从矿机交易市场、论坛和社区收集原始矿机描述
2. **数据清洗**：删除重复条目，修正明显错误
3. **数据标准化**：将非标准表述转换为统一格式
4. **数据验证**：确保所有标准化输出符合预期格式

## 使用方法

在训练过程中，可以直接使用`train.py`脚本加载此数据：

```python
from transformers import Trainer

# 训练数据路径已在 DataArguments 中设置为 datasets/train_data/train_v1.jsonl
trainer = Trainer(...)
trainer.train()
```

或者手动加载数据进行处理：

```python
import json

# 加载训练数据
with open('train_v1.jsonl', 'r', encoding='utf-8') as f:
    data = [json.loads(line) for line in f]

# 提取输入和输出
inputs = [item['input'] for item in data]
outputs = [item['output'] for item in data]
```

## 数据扩展

如需扩展数据集，请遵循以下规则添加新的训练样本：

1. 确保新添加的样本遵循相同的JSON格式
2. 标准化的输出应保持一致的格式：`矿机品牌 矿机系列 - 哈希率 价格`
3. 添加后验证JSONL文件格式是否正确

可以使用以下命令验证JSONL文件格式：

```bash
python -c "import json; [json.loads(l) for l in open('train_v1.jsonl', 'r')]"
```

## 注意事项

- 数据中的价格信息仅供训练使用，不代表实际市场价格
- 某些特殊型号可能存在多种标准化方式，模型训练后可能会选择其中一种作为输出
- 训练数据可能需要定期更新以包含新发布的矿机型号
