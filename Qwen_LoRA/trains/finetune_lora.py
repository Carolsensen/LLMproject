import json
import os
import torch
import argparse
import time
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    BitsAndBytesConfig,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer
from tqdm import tqdm
import pandas as pd

# 自定义数据收集器
class CustomDataCollator(DataCollatorForLanguageModeling):
    def torch_call(self, examples):
        # 过滤出包含text字段的样本
        valid_examples = [ex for ex in examples if "text" in ex and isinstance(ex["text"], str)]
        
        # 如果没有有效样本，创建一个占位批次
        if not valid_examples:
            print("警告: 所有样本都缺少text字段，创建占位批次")
            return {
                "input_ids": torch.zeros((1, 1), dtype=torch.long),
                "attention_mask": torch.ones((1, 1), dtype=torch.long),
                "labels": torch.zeros((1, 1), dtype=torch.long)
            }
        
        # 处理有效样本
        texts = [ex["text"] for ex in valid_examples]
        
        # 使用分词器处理文本
        batch = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=min(self.tokenizer.model_max_length, 2048),
            return_tensors="pt",  # 确保使用"pt"而不是"torch"
        )
        
        # 添加标签
        if "labels" not in batch:
            batch["labels"] = batch["input_ids"].clone()
        
        return batch

# 数据格式化函数
def format_dataset(example):
    messages = example["messages"]
    
    # 确保messages是列表
    if not isinstance(messages, list):
        messages = [messages]
    
    formatted_text = ""
    for message in messages:
        # 处理单个消息
        if isinstance(message, dict):
            role = message.get("role", "")
            content = message.get("content", "")
            
            # 处理特殊情况
            if role == "assistant" and content.startswith("/no_think"):
                content = content[9:].strip()
                
            if role and content:
                # 将role首字母大写，并添加冒号
                formatted_text += f"{role.capitalize()}: {content}\n"
        else:
            # 如果消息不是字典，尝试将其转换为字符串
            formatted_text += f"User: {str(message)}\n"
    
    # 添加结束标记
    formatted_text += tokenizer.eos_token
    
    return {"text": formatted_text}

class CustomTrainer(SFTTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.loss_history = []  # 存储loss历史数据
        self.start_time = time.time()  # 记录训练开始时间
        
    def compute_loss(self, model, inputs, return_outputs=False):
        # 确保输入格式正确
        if "input_ids" not in inputs or inputs["input_ids"].dim() != 2:
            print(f"警告: 输入input_ids格式不正确: {inputs.get('input_ids', '不存在')}")
            # 创建一个占位输入
            inputs["input_ids"] = torch.zeros((1, 1), dtype=torch.long).to(model.device)
            inputs["attention_mask"] = torch.ones((1, 1), dtype=torch.long).to(model.device)
        
        inputs["labels"] = inputs["input_ids"].clone()
        outputs = model(**inputs)
        loss = outputs.loss
        
        # 记录loss和时间
        step = len(self.loss_history) + 1
        elapsed_time = time.time() - self.start_time
        self.loss_history.append({
            "step": step,
            "loss": loss.item(),
            "time": elapsed_time
        })
        
        # 每10步保存一次loss数据
        if step % 10 == 0:
            self.save_loss_history()
            
        return (loss, outputs) if return_outputs else loss
    
    def save_loss_history(self):
        """保存loss历史数据到CSV文件"""
        df = pd.DataFrame(self.loss_history)
        loss_file = os.path.join(self.args.output_dir, "loss_history.csv")
        df.to_csv(loss_file, index=False)
        print(f"已保存loss数据到 {loss_file}")

# 解析命令行参数
def parse_args():
    parser = argparse.ArgumentParser(description="Finetune Qwen2.5 model with LoRA")
    parser.add_argument("--model_path", type=str, default="/mnt/workspace/qwen_model/Qwen/Qwen2.5-7B-Instruct",
                        help="Path to the base model")
    parser.add_argument("--data_path", type=str, default="/mnt/workspace/qwen_model/Qwen/Qwen2.5-7B-Instruct/tourism_data.jsonl",
                        help="Path to the training data")
    parser.add_argument("--output_dir", type=str, default="/mnt/workspace/qwen_model/lora_results",
                        help="Output directory for the finetuned model")
    parser.add_argument("--max_seq_length", type=int, default=512,
                        help="Maximum sequence length")
    parser.add_argument("--disable_packing", action="store_true",
                        help="Disable data packing")
    return parser.parse_args()

args = parse_args()

# 参数配置
MODEL_PATH = args.model_path
DATA_PATH = args.data_path
OUTPUT_DIR = args.output_dir

# 确保输出目录存在
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 量化配置
quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True
)

# 加载模型
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    quantization_config=quant_config,
    device_map="auto",
    trust_remote_code=True,
    low_cpu_mem_usage=True,
    torch_dtype=torch.float16,
)

# 加载tokenizer
tokenizer = AutoTokenizer.from_pretrained(
    MODEL_PATH,
    trust_remote_code=True
)
tokenizer.pad_token = tokenizer.eos_token

# LoRA配置
peft_config = LoraConfig(
    r=4,
    lora_alpha=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_dropout=0.1,
    bias="none",
    task_type="CAUSAL_LM",
)

# 准备数据集
dataset = load_dataset("json", data_files=DATA_PATH, split="train")

# 应用数据格式化
dataset = dataset.map(format_dataset, remove_columns=["messages"], num_proc=4)

# 数据验证
print(f"数据集大小: {len(dataset)}")
if len(dataset) > 0:
    print("数据字段:", dataset.column_names)
    
    # 检查样本
    sample = dataset[0]
    print(f"样本结构: {sample.keys()}")
    print(f"样本text类型: {type(sample['text'])}")
    print(f"样本text前100字符: {sample['text'][:100]}")
    
    # 尝试对单个样本进行分词
    try:
        encoded = tokenizer(sample['text'], return_tensors="pt")  # 确保使用"pt"
        print(f"编码后的输入形状: {encoded['input_ids'].shape}")
    except Exception as e:
        print(f"分词验证失败: {e}")
    
    # 验证数据收集器是否能处理样本
    try:
        collator = CustomDataCollator(tokenizer=tokenizer, mlm=False)
        batch = collator([sample])
        print(f"数据收集器测试成功，批次形状: {batch['input_ids'].shape}")
        
        # 额外验证：检查批次是否符合模型期望
        if "input_ids" in batch and batch["input_ids"].dim() != 2:
            print(f"警告: 批次input_ids维度不正确: {batch['input_ids'].dim()}")
    except Exception as e:
        print(f"数据收集器测试失败: {e}")
        # 打印详细的错误堆栈信息
        import traceback
        print(traceback.format_exc())

# 确保只保留text字段
dataset = dataset.remove_columns([col for col in dataset.column_names if col != "text"])

# 再次验证
print(f"处理后的字段: {dataset.column_names}")

# 训练参数 - 修复错误并优化
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=4,  # 增大批次大小以提高GPU利用率
    gradient_accumulation_steps=4,  # 减少梯度累积步数
    learning_rate=3e-5,
    num_train_epochs=1,
    logging_steps=10,  # 每10步输出一次日志
    save_strategy="steps",
    save_steps=50,  # 每50步保存一次模型
    fp16=True,
    remove_unused_columns=True,
    optim="adamw_torch_fused",  # 使用更高效的优化器
    weight_decay=0.01,
    max_grad_norm=0.3,
    warmup_ratio=0.05,
    lr_scheduler_type="cosine",
    report_to="tensorboard",  # 启用TensorBoard可视化
    dataloader_num_workers=4,  # 增加数据加载工作线程
    gradient_checkpointing=True,  # 启用梯度检查点以节省显存
    # 删除不支持的use_cache参数
    logging_dir=f"{OUTPUT_DIR}/logs",  # 日志目录
    disable_tqdm=False,  # 启用进度条
)

# 显式设置模型的use_cache参数
model.config.use_cache = False  # 与gradient_checkpointing兼容

# 创建自定义数据收集器
data_collator = CustomDataCollator(tokenizer=tokenizer, mlm=False)

# 创建Trainer
trainer = CustomTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    peft_config=peft_config,
    dataset_text_field="text",
    max_seq_length=args.max_seq_length,
    packing=not args.disable_packing,  # 启用数据打包以提高训练效率
    tokenizer=tokenizer,
    data_collator=data_collator
)

# 开始训练
try:
    trainer.train()
except KeyboardInterrupt:
    print("训练被手动中断，保存当前模型...")
finally:
    # 保存最终模型和loss历史
    trainer.save_model(f"{OUTPUT_DIR}/final_adapter")
    trainer.save_loss_history()
    print(f"训练完成，模型已保存到 {OUTPUT_DIR}")