# data_collator.py
from transformers import DataCollatorForLanguageModeling
import torch

class FixedDataCollator(DataCollatorForLanguageModeling):
    def __init__(self, tokenizer, mlm=False):  # 强制设置mlm=False
        super().__init__(tokenizer=tokenizer, mlm=mlm)
        
    def torch_call(self, examples):
        # 严格数据验证
        texts = []
        for ex in examples:
            if not isinstance(ex, dict) or "text" not in ex:
                raise ValueError(f"无效样本格式: {type(ex)}")
            if not isinstance(ex["text"], str):
                raise ValueError(f"text字段应为字符串，实际为{type(ex['text'])}")
            texts.append(ex["text"])
        
        batch = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=min(self.tokenizer.model_max_length, 2048),
            return_tensors="pt"
        )
        batch["labels"] = batch["input_ids"].clone()
        return batch