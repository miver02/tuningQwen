import logging
import os
import re
from fastapi import FastAPI, Query
import requests
import torch
import json
import argparse
import time
import glob
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel, PeftConfig
import traceback
import uvicorn


# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

class getModelName:
    def __init__(self):
        self.model_path = 'datasets/base_model'
        self.adapter_path = 'datasets/final_model'
        self.device = "cpu" if not torch.cuda.is_available() else "cuda"
        self.model, self.tokenizer = None, None

    def load_model(self):
        try:
            base_model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                trust_remote_code=True,
                device_map=self.device,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
            )
            logger.info(f"加载LoRA适配器: {self.model_path}")
            model = PeftModel.from_pretrained(base_model, self.adapter_path)
            logger.info("LoRA模型加载成功")

            logger.info("加载分词器...")
            # tokenizer_path = os.path.join(self.model_path, "tokenizer")
            tokenizer_path = self.model_path
            tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
            logger.info("模型加载完毕")
            return model, tokenizer
        except Exception as e:
            logger.error(f"模型加载失败：{e}")

    def generate_response(self, input_text):
        """生成模型回复"""
        try:
            # 构建Qwen1.5-Chat格式的对话
            conversation = f"<|im_start|>user\n我将为你提供一个矿机信息，你需要返回标准格式的矿机名称和规格，格式为：矿机品牌 矿机系列 - 哈希率 价格。\n\n矿机信息：{input_text}<|im_end|>\n<|im_start|>assistant\n"

            # 编码输入
            inputs = self.tokenizer(conversation, return_tensors="pt").to(self.model.device)
            
            # 使用模型生成输出
            start_time = time.time()
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=50,
                    do_sample=False,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.convert_tokens_to_ids("<|im_end|>")
                )
            generation_time = time.time() - start_time
            
            # 解码输出文本
            output_text = self.tokenizer.decode(outputs[0], skip_special_tokens=False)
            
            # 提取助手回复部分
            if "<|im_start|>assistant\n" in output_text:
                assistant_part = output_text.split("<|im_start|>assistant\n")[1]
                if "<|im_end|>" in assistant_part:
                    assistant_text = assistant_part.split("<|im_end|>")[0].strip()
                else:
                    assistant_text = assistant_part.strip()
            else:
                # 如果无法找到助手标记，返回完整输出
                logger.warning(f"警告：无法找到助手标记，返回完整输出")
                assistant_text = output_text

            response = {
                'simple_modelname':input_text,
                'generate_modelname':assistant_text,
                'generation_time':generation_time,
                'status':200
            }
            return assistant_text
        except Exception as e:
            logger.error(f"生成回复时出错: {str(e)}")
            traceback.print_exc()
            return None
        
    def main(self):
        try:
            logger.info(f"启动前检查")
            logger.info(f"微调模型的路径：{self.model_path}")
            logger.info(f"使用的设备: {self.device}")
            # 加载模型和分词器
            logger.info(f"开始加载模型")
            self.model, self.tokenizer = self.load_model()
        except Exception as e:
            logger.error(f"模型加载失败，无法继续: {str(e)}")
            return
        

getmodelname = None

@app.on_event("startup")
async def startup_event():
    """应用启动时初始化匹配器"""
    global getmodelname
    getmodelname = getModelName()
    getmodelname.main()

@app.get("/model_generate/get-model-name")
async def get_model_name(simple_name: str = Query()):
    global getmodelname
    generate_modelname = getmodelname.generate_response(simple_name)
    if generate_modelname is None:
        return {
            'status':404
        }
    origin_price = generate_modelname.split(' ')[-1]
    match = re.search(r'([\$￥])?(\d+\.?\d*)', origin_price)
    if match:
        price_unit = match.group(1)
        price = match.group(2)
    else:
        price=price_unit=None

    get_model_id_json = requests.get(f'http://47.251.15.52:8000/crawler/get-model-id?name={generate_modelname}')
    model_id_data = get_model_id_json.json()
    response = {
        'status':model_id_data.get('status', None),
        'model_id':model_id_data.get('model_id', None),
        'model_name':model_id_data.get('model_name', None),
        'price':price,
        'price_unit':price_unit
    }
    return response

if __name__ == "__main__":
    # 启动FastAPI服务器
    uvicorn.run(app, host="0.0.0.0", port=8000)