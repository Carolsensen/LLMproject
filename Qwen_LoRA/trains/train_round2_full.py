# train_round2_full.py
import torch
import pandas as pd
import os
from datetime import datetime
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    BitsAndBytesConfig,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig
from trl import SFTTrainer
from datasets import load_dataset

# ================= 配置区 =================
MODEL_PATH = "/mnt/workspace/qwen_model/Qwen/Qwen2.5-7B-Instruct"
DATA_PATH = "/mnt/workspace/qwen_model/Qwen/Qwen2.5-7B-Instruct/tourism_data.jsonl"
OUTPUT_DIR = "/mnt/workspace/qwen_model/round2_output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ================= 数据格式化 =================
def format_dataset(example):
    """严格确保输出纯文本"""
    messages = example["messages"]
    text = ""
    for msg in messages:
        if isinstance(msg, dict):
            content = str(msg.get("content", "")).strip()  # 强制转为字符串
            if msg.get("role") == "assistant" and content.startswith("/no_think"):
                content = content[9:]
            text += f"{msg.get('role', 'user').capitalize()}: {content}\n"
    return {"text": text + tokenizer.eos_token}  # 确保EOS标记

# ================= 安全的数据收集器 =================
class SafeDataCollator(DataCollatorForLanguageModeling):
    def __call__(self, examples):
        # 过滤无效样本
        texts = []
        for ex in examples:
            if isinstance(ex, dict) and "text" in ex and isinstance(ex["text"], str):
                texts.append(ex["text"])
            else:
                print(f"⚠️ 忽略无效样本: {type(ex)}")
        
        if not texts:
            raise ValueError("所有样本均无效，请检查数据格式！")
            
        # 强制启用padding和truncation
        batch = self.tokenizer(
            texts,
            padding="longest",
            truncation=True,
            max_length=1024,
            return_tensors="pt"
        )
        batch["labels"] = batch["input_ids"].clone()
        return batch

# ================= 自定义Trainer（添加loss记录） =================
class LossTrackingTrainer(SFTTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.loss_history = []  # 存储loss记录：[(step, loss, timestamp), ...]
        self.start_time = datetime.now()  # 记录训练开始时间

    def training_step(self, model, inputs):
        # 调用父类的训练步骤，获取loss
        loss = super().training_step(model, inputs)
        
        # 记录当前步数、loss和时间
        current_step = self.state.global_step  # 获取全局步数
        elapsed_time = (datetime.now() - self.start_time).total_seconds()
        self.loss_history.append({
            "step": current_step,
            "loss": loss.item(),
            "elapsed_time": elapsed_time
        })
        
        # 每10步（与logging_steps一致）临时保存一次
        if current_step % self.args.logging_steps == 0 and current_step > 0:
            self.save_loss_history()
        
        # 每100步（与save_steps一致）强制保存一次
        if current_step % self.args.save_steps == 0 and current_step > 0:
            self.save_loss_history()
        
        return loss

    def save_loss_history(self):
        """将loss历史保存到CSV文件"""
        loss_file = os.path.join(self.args.output_dir, "loss_history.csv")
        df = pd.DataFrame(self.loss_history)
        df.to_csv(loss_file, index=False)
        print(f"💾 已保存loss历史到 {loss_file}（最新步数: {self.state.global_step}）")

    def _save_checkpoint(self, *args, **kwargs):
        """保存checkpoint时额外确保loss历史被保存"""
        super()._save_checkpoint(*args, **kwargs)
        self.save_loss_history()

# ================= 初始化 =================
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token

# 加载并验证数据
dataset = load_dataset("json", data_files=DATA_PATH, split="train")
dataset = dataset.map(format_dataset, remove_columns=["messages"])
print("✅ 数据样本示例:", dataset[0]["text"][:100] + "...")

# ================= 训练配置 =================
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    learning_rate=5e-5,
    num_train_epochs=1,
    evaluation_strategy="no",
    save_strategy="steps",
    save_steps=100,  # 每100步保存checkpoint
    logging_steps=10,  # 每10步记录日志
    fp16=True,
    remove_unused_columns=False,
    optim="adamw_torch_fused",
    max_grad_norm=0.3,
    warmup_ratio=0.1,
    lr_scheduler_type="cosine",
    report_to="tensorboard",
    gradient_checkpointing=True,
    dataloader_num_workers=2,
)

# ================= 启动训练 =================
# 加载模型
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    device_map="auto",
    quantization_config=BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True
    ),
    trust_remote_code=True
)

# 使用自定义Trainer（带loss记录功能）
trainer = LossTrackingTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    peft_config=LoraConfig(
        r=8,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.05,
        task_type="CAUSAL_LM"
    ),
    max_seq_length=1024,
    dataset_text_field="text",
    tokenizer=tokenizer,
    data_collator=SafeDataCollator(tokenizer=tokenizer, mlm=False)
)

print("开始训练...")
trainer.train()
trainer.save_model(f"{OUTPUT_DIR}/final")
# 训练结束后最终保存一次loss历史
trainer.save_loss_history()