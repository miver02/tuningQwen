# This code is based on the revised code from fastchat based on tatsu-lab/stanford_alpaca.


from dataclasses import dataclass, field
import json
import math
import logging
import os
from typing import Dict, Optional, List
import torch
from torch.utils.data import Dataset
from deepspeed import zero
from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus
import transformers
import deepspeed
from transformers import Trainer
from transformers.trainer_pt_utils import LabelSmoother
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from accelerate.utils import DistributedType


IGNORE_TOKEN_ID = LabelSmoother.ignore_index


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="datasets/base_model")


@dataclass
class DataArguments:
    data_path: str = field(
        default="datasets/train_data/train_v1.jsonl", metadata={"help": "Path to the training data."}
    )
    eval_data_path: str = field(
        default=None, metadata={"help": "Path to the evaluation data."}
    )
    lazy_preprocess: bool = False


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(
        default=512,
        metadata={
            "help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )
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


@dataclass
class LoraArguments:
    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_target_modules: List[str] = field(
        default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    )
    lora_weight_path: str = ""
    lora_bias: str = "none"
    q_lora: bool = False


def maybe_zero_3(param):
    """在ZeRO-3下收集参数，或者在非ZeRO-3下直接返回参数副本"""
    try:
        if hasattr(param, "ds_id"):
            # 如果参数是DeepSpeed ZeRO-3管理的
            try:
                assert param.ds_status == ZeroParamStatus.NOT_AVAILABLE
                with zero.GatheredParameters([param]):
                    param_copy = param.data.detach().cpu().clone()
                return param_copy
            except Exception as e:
                print(f"ZeRO-3参数收集失败: {str(e)}, 尝试直接获取")
                return param.detach().cpu().clone()
        else:
            # 常规参数
            return param.detach().cpu().clone()
    except Exception as e:
        print(f"处理参数时出错: {str(e)}, 尝试直接返回原参数")
        try:
            return param.detach().cpu().clone()
        except:
            return param  # 最后的备选，直接返回原参数


# Borrowed from peft.utils.get_peft_model_state_dict
def get_peft_state_maybe_zero_3(named_params, bias):
    """获取PEFT/LoRA模型的state_dict，处理可能的ZeRO-3参数"""
    try:
        if bias == "none":
            to_return = {k: t for k, t in named_params if "lora_" in k}
        elif bias == "all":
            to_return = {k: t for k, t in named_params if "lora_" in k or "bias" in k}
        elif bias == "lora_only":
            to_return = {}
            maybe_lora_bias = {}
            lora_bias_names = set()
            for k, t in named_params:
                if "lora_" in k:
                    to_return[k] = t
                    bias_name = k.split("lora_")[0] + "bias"
                    lora_bias_names.add(bias_name)
                elif "bias" in k:
                    maybe_lora_bias[k] = t
            for bias_name in lora_bias_names:
                if bias_name in maybe_lora_bias:
                    to_return[bias_name] = maybe_lora_bias[bias_name]
        else:
            raise NotImplementedError(f"未实现的bias类型: {bias}")
        
        # 处理ZeRO-3参数
        print(f"处理LoRA参数，共 {len(to_return)} 个参数")
        processed_params = {}
        for k, v in to_return.items():
            try:
                processed_params[k] = maybe_zero_3(v)
            except Exception as e:
                print(f"处理参数 {k} 时出错: {str(e)}, 跳过")
                continue
        
        return processed_params
    except Exception as e:
        print(f"获取PEFT状态时出错: {str(e)}")
        # 返回一个空的state_dict
        return {}


local_rank = None

def rank0_print(*args):
    if local_rank == 0:
        print(*args)


def safe_save_model_for_hf_trainer(trainer: transformers.Trainer, output_dir: str, bias="none"):
    """Collects the state dict and dump to disk."""
    # check if zero3 mode enabled
    is_zero3_enabled = False
    # 检查是否使用了DeepSpeed ZeRO-3
    try:
        if hasattr(trainer.accelerator.state, "deepspeed_plugin"):
            ds_plugin = trainer.accelerator.state.deepspeed_plugin
            is_zero3_enabled = ds_plugin.zero_stage == 3 if ds_plugin is not None else False
        print(f"是否使用了DeepSpeed ZeRO-3: {is_zero3_enabled}")
    except:
        print("无法检测DeepSpeed ZeRO-3状态，假设未启用")
    
    if is_zero3_enabled:
        # 如果启用了ZeRO-3
        try:
            state_dict = trainer.model_wrapped._zero3_consolidated_16bit_state_dict()
            print("已获取ZeRO-3整合的state_dict")
        except Exception as e:
            print(f"获取ZeRO-3整合state_dict失败: {str(e)}, 尝试使用一般方法")
            state_dict = trainer.model.state_dict()
    else:
        if trainer.args.use_lora:
            try:
                state_dict = get_peft_state_maybe_zero_3(
                    trainer.model.named_parameters(), bias
                )
                print("已获取LoRA state_dict")
            except Exception as e:
                print(f"获取LoRA state_dict失败: {str(e)}, 尝试使用一般方法")
                state_dict = trainer.model.state_dict()
        else:
            state_dict = trainer.model.state_dict()
    
    if trainer.args.should_save and trainer.args.local_rank == 0:
        try:
            trainer._save(output_dir, state_dict=state_dict)
            print(f"模型state_dict已保存到: {output_dir}")
        except Exception as e:
            print(f"保存state_dict时出错: {str(e)}")
            # 尝试直接使用save_pretrained方法
            try:
                print("尝试使用save_pretrained方法保存模型")
                trainer.model.save_pretrained(output_dir)
                print(f"使用save_pretrained成功保存模型到: {output_dir}")
            except Exception as e2:
                print(f"使用save_pretrained保存失败: {str(e2)}")


def preprocess(
    sources,
    tokenizer: transformers.PreTrainedTokenizer,
    max_len: int,
) -> Dict:
    input_ids, labels = [], []
    for example in sources:
        input_text = example["input"]
        output_text = example["output"]
        
        # 构建Qwen1.5-Chat格式的对话
        conversation = f"<|im_start|>user\n我将为你提供一个矿机信息，你需要返回标准格式的矿机名称和规格，格式为：矿机品牌 矿机系列 - 哈希率 价格。\n\n矿机信息：{input_text}<|im_end|>\n<|im_start|>assistant\n{output_text}<|im_end|>"
        
        # 对整个对话编码
        encoded = tokenizer.encode(conversation, add_special_tokens=False)
        
        # 寻找助手回复的起始和结束位置
        assistant_start_str = "<|im_start|>assistant\n"
        try:
            assistant_start_pos = conversation.find(assistant_start_str)
            if assistant_start_pos == -1:
                # 如果找不到助手标记，使用默认值
                print(f"警告：无法找到助手标记在样本中: {conversation[:50]}...")
                # 将所有标记视为忽略
                label = [-100] * len(encoded)
            else:
                assistant_start_pos += len(assistant_start_str)
                assistant_start_token_pos = len(tokenizer.encode(conversation[:assistant_start_pos], add_special_tokens=False))
                
                # 创建输入和标签
                label = [-100] * assistant_start_token_pos + encoded[assistant_start_token_pos:]
        except Exception as e:
            print(f"处理样本时出错: {str(e)}")
            # 将所有标记视为忽略
            label = [-100] * len(encoded)
        
        input_id = encoded
        
        # 截断或填充
        if len(input_id) > max_len:
            input_id = input_id[:max_len]
            label = label[:max_len]
        else:
            pad_len = max_len - len(input_id)
            input_id = input_id + [tokenizer.pad_token_id] * pad_len
            label = label + [-100] * pad_len
            
        input_ids.append(input_id)
        labels.append(label)
    
    input_ids = torch.tensor(input_ids, dtype=torch.long)
    labels = torch.tensor(labels, dtype=torch.long)

    return dict(
        input_ids=input_ids,
        labels=labels,
        attention_mask=input_ids.ne(tokenizer.pad_token_id),
    )


class SupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, raw_data, tokenizer: transformers.PreTrainedTokenizer, max_len: int):
        super(SupervisedDataset, self).__init__()

        rank0_print("Formatting inputs...")
        self.tokenizer = tokenizer
        self.max_len = max_len
        data_dict = preprocess(raw_data, tokenizer, max_len)

        self.input_ids = data_dict["input_ids"]
        self.labels = data_dict["labels"]
        self.attention_mask = data_dict["attention_mask"]

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        return dict(
            input_ids=self.input_ids[i],
            labels=self.labels[i],
            attention_mask=self.attention_mask[i],
        )


class LazySupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, raw_data, tokenizer: transformers.PreTrainedTokenizer, max_len: int):
        super(LazySupervisedDataset, self).__init__()
        self.tokenizer = tokenizer
        self.max_len = max_len

        rank0_print("Formatting inputs...Skip in lazy mode")
        self.raw_data = raw_data
        self.cached_data_dict = {}

    def __len__(self):
        return len(self.raw_data)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        if i in self.cached_data_dict:
            return self.cached_data_dict[i]

        ret = preprocess([self.raw_data[i]], self.tokenizer, self.max_len)
        ret = dict(
            input_ids=ret["input_ids"][0],
            labels=ret["labels"][0],
            attention_mask=ret["attention_mask"][0],
        )
        self.cached_data_dict[i] = ret

        return ret


def make_supervised_data_module(
    tokenizer: transformers.PreTrainedTokenizer, data_args, max_len,
) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    dataset_cls = (
        LazySupervisedDataset if data_args.lazy_preprocess else SupervisedDataset
    )
    rank0_print("Loading data...")

    # 加载数据
    raw_data = []
    try:
        print(f"正在从 {data_args.data_path} 加载训练数据...")
        with open(data_args.data_path, "r", encoding="utf-8") as f:
            line_count = 0
            for line in f:
                line_count += 1
                try:
                    example = json.loads(line.strip())
                    if "input" not in example or "output" not in example:
                        print(f"警告: 第 {line_count} 行数据格式不正确，跳过")
                        continue
                    raw_data.append(example)
                except json.JSONDecodeError:
                    print(f"警告: 第 {line_count} 行不是有效的JSON，跳过")
                    continue
        
        print(f"成功加载 {len(raw_data)} 条训练数据")
        
        # 打印几个样本示例
        if len(raw_data) > 0:
            print("数据示例:")
            for i, example in enumerate(raw_data[:3]):
                print(f"示例 {i+1}:")
                print(f"  输入: {example['input']}")
                print(f"  输出: {example['output']}")
                print()
    except Exception as e:
        print(f"加载训练数据失败: {str(e)}")
        raise
    
    train_dataset = dataset_cls(raw_data, tokenizer=tokenizer, max_len=max_len)

    if data_args.eval_data_path:
        try:
            print(f"正在从 {data_args.eval_data_path} 加载评估数据...")
            eval_raw_data = []
            with open(data_args.eval_data_path, "r", encoding="utf-8") as f:
                line_count = 0
                for line in f:
                    line_count += 1
                    try:
                        example = json.loads(line.strip())
                        if "input" not in example or "output" not in example:
                            print(f"警告: 评估数据第 {line_count} 行格式不正确，跳过")
                            continue
                        eval_raw_data.append(example)
                    except json.JSONDecodeError:
                        print(f"警告: 评估数据第 {line_count} 行不是有效的JSON，跳过")
                        continue
            
            print(f"成功加载 {len(eval_raw_data)} 条评估数据")
            eval_dataset = dataset_cls(eval_raw_data, tokenizer=tokenizer, max_len=max_len)
        except Exception as e:
            print(f"加载评估数据失败: {str(e)}")
            print("将不使用评估数据")
            eval_dataset = None
    else:
        eval_dataset = None

    return dict(train_dataset=train_dataset, eval_dataset=eval_dataset)


def train():
    global local_rank

    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments, LoraArguments)
    )
    (
        model_args,
        data_args,
        training_args,
        lora_args,
    ) = parser.parse_args_into_dataclasses()

    # This serves for single-gpu qlora.
    try:
        if getattr(training_args, 'deepspeed', None) and int(os.environ.get("WORLD_SIZE", 1))==1:
            training_args.distributed_state.distributed_type = DistributedType.DEEPSPEED
            print("已设置单GPU DeepSpeed模式")
    except Exception as e:
        print(f"设置DeepSpeed模式时出错: {str(e)}")

    local_rank = training_args.local_rank

    device_map = None
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    ddp = world_size != 1
    if lora_args.q_lora:
        device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)} if ddp else "auto"
        try:
            # 检查是否使用了ZeRO-3或FSDP
            is_zero3_enabled = False
            if hasattr(training_args, 'fsdp') and len(training_args.fsdp) > 0:
                print("检测到使用了FSDP")
                is_zero3_enabled = False
            
            if hasattr(training_args.accelerator.state, "deepspeed_plugin"):
                ds_plugin = training_args.accelerator.state.deepspeed_plugin
                is_zero3_enabled = ds_plugin.zero_stage == 3 if ds_plugin is not None else False
                if is_zero3_enabled:
                    print("检测到使用了DeepSpeed ZeRO-3")
            
            if is_zero3_enabled or len(training_args.fsdp) > 0:
                logging.warning(
                    "FSDP或ZeRO3与QLoRA不兼容，可能会导致问题。"
                )
        except Exception as e:
            print(f"检查ZeRO-3/FSDP状态时出错: {str(e)}")

    # 设置配置
    try:
        config = transformers.AutoConfig.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            trust_remote_code=True,
        )
        config.use_cache = False
        
        # 打印模型基本信息
        print(f"加载模型：{model_args.model_name_or_path}")
        print(f"模型类型：{config.model_type}")
        if hasattr(config, "num_hidden_layers"):
            print(f"模型层数：{config.num_hidden_layers}")

        # 加载模型和分词器
        model = transformers.AutoModelForCausalLM.from_pretrained(
            model_args.model_name_or_path,
            config=config,
            cache_dir=training_args.cache_dir,
            device_map=device_map,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
        )
        
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            model_max_length=training_args.model_max_length,
            padding_side="right",
            use_fast=False,
            trust_remote_code=True,
        )
        
        print(f"模型和分词器加载成功!")
        
    except Exception as e:
        print(f"加载模型或分词器时出错: {str(e)}")
        raise
    
    # 确保padding token正确设置
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
        print(f"设置pad_token_id = {tokenizer.pad_token_id} (eos_token_id)")
    
    # 打印特殊token信息
    try:
        special_tokens = {
            "pad_token_id": tokenizer.pad_token_id,
            "eos_token_id": tokenizer.eos_token_id,
            "bos_token_id": tokenizer.bos_token_id if hasattr(tokenizer, 'bos_token_id') else None,
            "unk_token_id": tokenizer.unk_token_id if hasattr(tokenizer, 'unk_token_id') else None
        }
        print(f"特殊token信息: {special_tokens}")
    except:
        print("获取特殊token信息失败")

    # 如果使用LoRA进行微调
    if training_args.use_lora:
        lora_config = LoraConfig(
            r=lora_args.lora_r,
            lora_alpha=lora_args.lora_alpha,
            target_modules=lora_args.lora_target_modules,
            lora_dropout=lora_args.lora_dropout,
            bias=lora_args.lora_bias,
            task_type="CAUSAL_LM",
        )
        
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()
        
        if training_args.gradient_checkpointing:
            model.enable_input_require_grads()

    # 加载数据
    data_module = make_supervised_data_module(
        tokenizer=tokenizer, data_args=data_args, max_len=training_args.model_max_length
    )

    # 启动训练
    trainer = Trainer(
        model=model, tokenizer=tokenizer, args=training_args, **data_module
    )

    trainer.train()
    trainer.save_state()

    # 保存模型
    try:
        safe_save_model_for_hf_trainer(trainer=trainer, output_dir=training_args.output_dir, bias=lora_args.lora_bias)
        print("已保存模型的state_dict")
    except Exception as e:
        print(f"保存模型state_dict时出错: {str(e)}")
    
    # 确保在任何情况下都保存分词器
    try:
        tokenizer_save_path = os.path.join(training_args.output_dir, "tokenizer")
        os.makedirs(tokenizer_save_path, exist_ok=True)
        tokenizer.save_pretrained(training_args.output_dir)
        print(f"分词器已保存到: {training_args.output_dir}")
    except Exception as e:
        print(f"保存分词器时出错: {str(e)}")
    
    # 保存完整模型（非LoRA）
    if training_args.use_lora:
        try:
            # 合并LoRA权重并保存完整模型
            print("正在合并LoRA权重并保存完整模型...")
            merged_model = model.merge_and_unload()
            save_path = os.path.join(training_args.output_dir, "merged_model")
            os.makedirs(save_path, exist_ok=True)
            merged_model.save_pretrained(save_path)
            print(f"完整合并模型保存成功！路径: {save_path}")
            
            # 再次保存原始LoRA模型，以防万一
            lora_save_path = os.path.join(training_args.output_dir, "lora_model")
            os.makedirs(lora_save_path, exist_ok=True)
            model.save_pretrained(lora_save_path)
            print(f"LoRA模型权重保存成功！路径: {lora_save_path}")
        except Exception as e:
            print(f"警告：合并LoRA权重失败：{str(e)}")
            print("将仅保存LoRA模型权重")
            try:
                # 保存LoRA模型
                model.save_pretrained(training_args.output_dir)
                print(f"LoRA模型权重保存成功！路径: {training_args.output_dir}")
            except Exception as e2:
                print(f"保存LoRA模型权重也失败: {str(e2)}")
    else:
        try:
            model.save_pretrained(training_args.output_dir)
            print(f"模型已保存到: {training_args.output_dir}")
        except Exception as e:
            print(f"保存模型时出错: {str(e)}")

    print("训练和保存过程完成！")


if __name__ == "__main__":
    train()