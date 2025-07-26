# manual_check.py
# manual_check_fixed.py
import torch
import json
from transformers import AutoModelForCausalLM, AutoTokenizer

# 配置路径
MODEL_PATH = "/mnt/workspace/qwen_model/Qwen/Qwen2.5-7B-Instruct"

# 加载模型和tokenizer
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    device_map="auto",
    torch_dtype=torch.float16
)
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)

# 测试问题
test_question = "推荐兰州三日游行程"
inputs = tokenizer(test_question, return_tensors="pt").to("cuda")
outputs = model.generate(**inputs, max_new_tokens=200)
print("模型输出:\n", tokenizer.decode(outputs[0], skip_special_tokens=True))
