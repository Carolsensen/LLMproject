# compare_model.py
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# 初始化基础模型和tokenizer（未微调）
base_model = AutoModelForCausalLM.from_pretrained(
    "/mnt/workspace/qwen_model/Qwen/Qwen2.5-7B-Instruct",
    device_map="auto",
    torch_dtype=torch.float16
)
tokenizer = AutoTokenizer.from_pretrained(
    "/mnt/workspace/qwen_model/Qwen/Qwen2.5-7B-Instruct",
    trust_remote_code=True
)

# 加载微调后的LoRA适配器
tuned_model = PeftModel.from_pretrained(base_model, "/mnt/workspace/qwen_model/lora_results/final_adapter")

# 测试问题集（覆盖训练数据中的城市/场景）
test_questions = [
    "用JSON格式生成兰州三日游行程，包含交通和餐饮建议",
    "张掖丹霞地貌的最佳游览时间是什么时候？请说明理由",
    "推荐三家西安性价比高的四星级酒店，要求靠近地铁站"
]

def generate_answer(model, question):
    inputs = tokenizer(question, return_tensors="pt").to("cuda")
    outputs = model.generate(**inputs, max_new_tokens=200)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# 对比测试
for q in test_questions:
    print(f"\n问题：{q}")
    print("原始模型回答：\n" + generate_answer(base_model, q))
    print("微调模型回答：\n" + generate_answer(tuned_model, q))
    print("="*80)