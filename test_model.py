import torch
import json
import argparse
import os
import time
import glob
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel, PeftConfig
import traceback

def generate_response(model, tokenizer, input_text):
    """生成模型回复"""
    try:
        # 构建Qwen1.5-Chat格式的对话
        conversation = f"<|im_start|>user\n我将为你提供一个矿机信息，你需要返回标准格式的矿机名称和规格，格式为：矿机品牌 矿机系列 - 哈希率 价格。\n\n矿机信息：{input_text}<|im_end|>\n<|im_start|>assistant\n"

        # 编码输入
        inputs = tokenizer(conversation, return_tensors="pt").to(model.device)
        
        # 使用模型生成输出
        start_time = time.time()
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=50,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.convert_tokens_to_ids("<|im_end|>")
            )
        generation_time = time.time() - start_time
        
        # 解码输出文本
        output_text = tokenizer.decode(outputs[0], skip_special_tokens=False)
        
        # 提取助手回复部分
        if "<|im_start|>assistant\n" in output_text:
            assistant_part = output_text.split("<|im_start|>assistant\n")[1]
            if "<|im_end|>" in assistant_part:
                assistant_text = assistant_part.split("<|im_end|>")[0].strip()
            else:
                assistant_text = assistant_part.strip()
        else:
            # 如果无法找到助手标记，返回完整输出
            print(f"警告：无法找到助手标记，返回完整输出")
            assistant_text = output_text
        
        return assistant_text, generation_time
    except Exception as e:
        print(f"生成回复时出错: {str(e)}")
        traceback.print_exc()
        return f"生成错误: {str(e)}", 0

def load_model(model_path, device, use_lora=False, base_model_path=None):
    """加载模型，支持LoRA模型和完整模型"""
    try:
        # 检查模型路径下是否有不同的子目录
        has_merged_model = os.path.exists(os.path.join(model_path, "merged_model"))
        has_lora_model = os.path.exists(os.path.join(model_path, "lora_model"))
        is_lora_adapter = os.path.exists(os.path.join(model_path, "adapter_config.json"))
        
        # 打印检测到的模型类型
        print(f"检测到的模型类型:")
        print(f"- 合并模型: {has_merged_model}")
        print(f"- LoRA模型: {has_lora_model}")
        print(f"- 是LoRA适配器: {is_lora_adapter}")
        
        # 根据不同的模型类型进行加载
        if use_lora or is_lora_adapter:
            # 如果是LoRA模型
            if base_model_path is None:
                base_model_path = "datasets/base_model"
                print(f"未指定基础模型路径，使用默认路径: {base_model_path}")
            
            print(f"加载基础模型: {base_model_path}")
            base_model = AutoModelForCausalLM.from_pretrained(
                base_model_path,
                trust_remote_code=True,
                device_map=device,
                torch_dtype=torch.float16 if device == "cuda" else torch.float32
            )
            
            print(f"加载LoRA适配器: {model_path}")
            model = PeftModel.from_pretrained(base_model, model_path)
            print("LoRA模型加载成功")
            
        elif has_merged_model:
            # 使用合并后的模型
            merged_path = os.path.join(model_path, "merged_model")
            print(f"使用合并模型路径: {merged_path}")
            model = AutoModelForCausalLM.from_pretrained(
                merged_path,
                trust_remote_code=True,
                device_map=device,
                torch_dtype=torch.float16 if device == "cuda" else torch.float32
            )
            print("合并模型加载成功")
        else:
            # 使用普通模型
            print(f"加载普通模型: {model_path}")
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                trust_remote_code=True,
                device_map=device,
                torch_dtype=torch.float16 if device == "cuda" else torch.float32
            )
            print("模型加载成功")
        
        # 加载分词器
        print("加载分词器...")
        # 尝试从不同路径加载分词器
        tokenizer_paths = [
            model_path,
            os.path.join(model_path, "tokenizer"),
            base_model_path if base_model_path else None
        ]
        
        tokenizer = None
        for path in tokenizer_paths:
            if path:
                try:
                    tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
                    print(f"从 {path} 成功加载分词器")
                    break
                except Exception as e:
                    print(f"从 {path} 加载分词器失败: {str(e)}")
        
        if tokenizer is None:
            raise ValueError("无法加载分词器")
        
        # 确保padding token正确设置
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = tokenizer.eos_token_id
            print(f"设置pad_token_id = {tokenizer.pad_token_id} (eos_token_id)")
        
        model = model.eval()
        print(f"模型大小: {sum(p.numel() for p in model.parameters())/1000000:.2f}M 参数")
        
        return model, tokenizer
    except Exception as e:
        print(f"加载模型失败: {str(e)}")
        traceback.print_exc()
        raise

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="datasets/final_model", help="训练好的模型路径")
    parser.add_argument("--base_model_path", type=str, default=None, help="当使用LoRA模型时的基础模型路径")
    parser.add_argument("--test_file", type=str, default="datasets/train_data/train_v1.jsonl", help="测试数据文件")
    parser.add_argument("--num_samples", type=int, default=5, help="测试样本数量")
    parser.add_argument("--use_cpu", action="store_true", help="强制使用CPU")
    parser.add_argument("--use_lora", action="store_true", help="强制使用LoRA模式加载")
    parser.add_argument("--output_file", type=str, default="test_results.jsonl", help="测试结果输出文件")
    args = parser.parse_args()
    
    # 设置设备
    device = "cpu" if args.use_cpu or not torch.cuda.is_available() else "cuda"
    print(f"使用设备: {device}")
    
    # 检查模型路径
    if not os.path.exists(args.model_path):
        print(f"错误: 模型路径不存在: {args.model_path}")
        return
    
    # 检查测试文件
    if not os.path.exists(args.test_file):
        print(f"错误: 测试文件不存在: {args.test_file}")
        return
    
    try:
        # 加载模型和分词器
        model, tokenizer = load_model(
            args.model_path, 
            device,
            use_lora=args.use_lora,
            base_model_path=args.base_model_path
        )
    except Exception as e:
        print(f"模型加载失败，无法继续: {str(e)}")
        return
    
    # 加载测试数据
    test_samples = []
    try:
        print(f"正在加载测试数据: {args.test_file}")
        with open(args.test_file, 'r', encoding='utf-8') as f:
            data = [json.loads(line) for line in f]
            
            # 随机选择一些样本
            import random
            if len(data) > args.num_samples:
                test_samples = random.sample(data, args.num_samples)
            else:
                test_samples = data
        
        print(f"加载了 {len(test_samples)} 个测试样本")
    except Exception as e:
        print(f"加载测试数据失败: {str(e)}")
        traceback.print_exc()
        return
    
    # 进行测试
    correct = 0
    total_time = 0
    all_results = []
    
    for i, sample in enumerate(test_samples):
        try:
            input_text = sample["input"]
            expected_output = sample["output"]
            
            print(f"\n样本 {i+1}:")
            print(f"输入: {input_text}")
            print(f"预期输出: {expected_output}")
            
            # 生成模型回复
            generated_output, gen_time = generate_response(model, tokenizer, input_text)
            total_time += gen_time
            print(f"模型输出: {generated_output}")
            print(f"生成时间: {gen_time:.2f}秒")
            
            # 简单评估结果
            is_correct = generated_output.strip() == expected_output.strip()
            if is_correct:
                correct += 1
                print("结果匹配 ✓")
            else:
                print("结果不匹配 ✗")
            
            # 收集结果
            all_results.append({
                "input": input_text,
                "expected": expected_output,
                "generated": generated_output,
                "correct": is_correct,
                "time": gen_time
            })
        except Exception as e:
            print(f"处理样本时出错: {str(e)}")
            traceback.print_exc()
    
    # 报告结果
    if test_samples:
        accuracy = correct / len(test_samples)
        avg_time = total_time / len(test_samples)
        print(f"\n测试完成，准确率: {accuracy:.2%} ({correct}/{len(test_samples)})")
        print(f"平均生成时间: {avg_time:.2f}秒/样本")
        
        # 保存测试结果
        if args.output_file:
            try:
                with open(args.output_file, 'w', encoding='utf-8') as f:
                    f.write(json.dumps({
                        "accuracy": accuracy,
                        "correct": correct,
                        "total": len(test_samples),
                        "avg_time": avg_time,
                        "results": all_results
                    }, ensure_ascii=False, indent=2))
                print(f"测试结果已保存到: {args.output_file}")
            except Exception as e:
                print(f"保存测试结果失败: {str(e)}")
    else:
        print("没有测试样本")

if __name__ == "__main__":
    main() 