# simple_eval.py
import torch
import json
import re
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# 手动配置路径
TEST_FILE = "cleaned_test_v2.jsonl"
MODEL_PATH = "/mnt/workspace/qwen_model/Qwen/Qwen2.5-7B-Instruct"
LORA_PATH = "/mnt/workspace/qwen_model/lora_results/final_adapter"

# 加载测试数据
with open(TEST_FILE) as f:
    test_data = [json.loads(line) for line in f]

# 加载模型
base_model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    device_map="auto",
    torch_dtype=torch.float16
)
tuned_model = PeftModel.from_pretrained(base_model, LORA_PATH)
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

# 简易评估
for item in test_data[:3]:  # 只评估前3条
    inputs = tokenizer(item["question"], return_tensors="pt").to("cuda")
    outputs = tuned_model.generate(**inputs, max_new_tokens=200)
    print(f"\n问题：{item['question']}\n回答：{tokenizer.decode(outputs[0], skip_special_tokens=True)}")