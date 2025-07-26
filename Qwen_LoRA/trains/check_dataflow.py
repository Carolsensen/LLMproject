# check_dataflow.py
# check_dataflow_fixed.py
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from data_collator import FixedDataCollator  # 需单独保存

# 配置参数
DATA_PATH = "/mnt/workspace/qwen_model/Qwen/Qwen2.5-7B-Instruct/tourism_data.jsonl"
MODEL_PATH = "/mnt/workspace/qwen_model/Qwen/Qwen2.5-7B-Instruct"

def format_dataset(example):
    """修正后的数据格式化函数"""
    messages = example["messages"]
    formatted_text = ""
    for msg in messages:
        if isinstance(msg, dict):
            content = msg["content"]
            if msg["role"] == "assistant" and content.startswith("/no_think"):
                content = content[9:].strip()
            formatted_text += f"{msg['role'].capitalize()}: {content}\n"
    formatted_text += tokenizer.eos_token
    return {"text": formatted_text}

# 初始化
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token

# 1. 数据加载测试
try:
    dataset = load_dataset("json", data_files=DATA_PATH, split="train")
    dataset = dataset.map(format_dataset, remove_columns=["messages"])
    print("✅ 数据加载成功")
    print("样本字段:", dataset.column_names)
    print("首样本text内容:", dataset[0]["text"][:100] + "...")
except Exception as e:
    print(f"❌ 数据加载失败: {str(e)}")
    exit()

# 2. 数据收集器测试
try:
    collator = FixedDataCollator(tokenizer=tokenizer, mlm=False)  # 关键修正：添加mlm=False
    batch = collator([dataset[0], dataset[1]])
    
    print("\n🔍 批次检查:")
    print("input_ids形状:", batch["input_ids"].shape)
    print("input_ids示例:", batch["input_ids"][0][:10])
    print("labels与input_ids一致?", torch.all(batch["input_ids"] == batch["labels"]))
    
    if batch["input_ids"].sum().item() == 0:
        print("❌ 警告：批次输入全为零！")
    else:
        print("✅ 批次数据有效")
        
    # GPU测试
    device_batch = {k: v.to("cuda") for k, v in batch.items()}
    print("✅ GPU数据传输成功")
except Exception as e:
    print(f"❌ 数据收集器测试失败: {str(e)}")
    exit()

# 3. 模型前向传播测试
try:
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        device_map="auto",
        torch_dtype=torch.float16
    )
    outputs = model(**device_batch)  # 使用GPU批次
    loss = outputs.loss
    print(f"\n🎯 前向传播测试: loss={loss.item():.4f}")
    if loss.item() == 0:
        print("❌ 警告：Loss为零，请检查数据！")
except Exception as e:
    print(f"❌ 模型测试失败: {str(e)}")