# eval_metrics_fixed_v2.py
# eval_metrics_fixed_v3.py
import json
import torch
from rouge_score import rouge_scorer
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# 配置路径
MODEL_PATH = "/mnt/workspace/qwen_model/Qwen/Qwen2.5-7B-Instruct"
LORA_PATH = "/mnt/workspace/qwen_model/lora_results/final_adapter"
TEST_DATA_PATH = "tourism_test_fixed.jsonl"

# 加载测试数据
with open(TEST_DATA_PATH) as f:
    test_data = [json.loads(line) for line in f][:5]  # 只测试5条加速

# 初始化评分器
scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)

def evaluate(model, tokenizer):
    scores = []
    for item in test_data:
        inputs = tokenizer(item["question"], return_tensors="pt", truncation=True, max_length=512).to("cuda")
        outputs = model.generate(**inputs, max_new_tokens=100)
        pred = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        print(f"\n问题: {item['question']}")
        print(f"参考答案: {item['answer'][:100]}...")
        print(f"模型输出: {pred[:100]}...")
        
        scores.append(scorer.score(item["answer"], pred)["rougeL"].fmeasure)
        torch.cuda.empty_cache()
    return scores

# 加载基础模型
base_model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    device_map="auto",
    torch_dtype=torch.float16
)
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)

# 评估原始模型
print("\n评估原始模型...")
base_scores = evaluate(base_model, tokenizer)
print(f"\n原始模型 Rouge-L 平均分: {sum(base_scores)/len(base_scores):.3f}")

# 加载微调模型
tuned_model = PeftModel.from_pretrained(base_model, LORA_PATH)

# 评估微调模型
print("\n评估微调模型...")
tuned_scores = evaluate(tuned_model, tokenizer)
print(f"\n微调模型 Rouge-L 平均分: {sum(tuned_scores)/len(tuned_scores):.3f}")

# 计算提升比例
if sum(base_scores) > 0:
    improvement = (sum(tuned_scores) - sum(base_scores)) / sum(base_scores) * 100
    print(f"\n性能提升: {improvement:.1f}%")
else:
    print("\n警告：基础模型得分为0，无法计算提升比例")