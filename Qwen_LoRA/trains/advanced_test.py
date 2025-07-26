# advanced_test.py
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# 初始化模型和tokenizer
MODEL_PATH = "/mnt/workspace/qwen_model/Qwen/Qwen2.5-7B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    device_map="auto",
    torch_dtype=torch.float16,
    trust_remote_code=True
).eval()

# 测试问题列表
questions = [
    "用JSON格式输出兰州三日游行程",
    "张掖丹霞地貌的最佳游览季节是？", 
    "推荐西安性价比高的四星级酒店"
]

# 执行测试
for q in questions:
    inputs = tokenizer(q, return_tensors="pt").to("cuda")
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=200)
    print(f"\n问题：{q}\n回答：{tokenizer.decode(outputs[0], skip_special_tokens=True)}")
    print("="*50)