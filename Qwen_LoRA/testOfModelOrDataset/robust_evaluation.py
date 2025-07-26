# robust_evaluation.py
import json
import torch
from rouge_score import rouge_scorer
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# 配置
MODEL_PATH = "/mnt/workspace/qwen_model/Qwen/Qwen2.5-7B-Instruct"
#LORA_PATH = "/mnt/workspace/qwen_model/lora_results/final_adapter"
LORA_PATH = "/mnt/workspace/qwen_model/round2_output/final"
TEST_FILE = "tourism_test_valid.jsonl"

# 初始化
device = "cuda" if torch.cuda.is_available() else "cpu"
scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)

# 加载模型
base_model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    device_map="auto",
    torch_dtype=torch.float16
).eval()
tuned_model = PeftModel.from_pretrained(base_model, LORA_PATH).eval()

def evaluate(model, data):
    scores = []
    for item in data[:20]:  # 评估前20条加速测试
        try:
            # 生成回答
            inputs = tokenizer(item["question"], return_tensors="pt").to(device)
            outputs = model.generate(**inputs, max_new_tokens=200)
            pred = tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # 计算ROUGE-L
            score = scorer.score(item["answer"], pred)["rougeL"].fmeasure
            scores.append(score)
            
            # 打印样本
            print(f"\nQ: {item['question']}\nA: {item['answer']}\nModel: {pred[:100]}...\nScore: {score:.3f}")
        except Exception as e:
            print(f"处理失败: {str(e)}")
            scores.append(0)
    return scores

# 加载测试数据
with open(TEST_FILE) as f:
    test_data = [json.loads(line) for line in f]

# 评估
print("=== 原始模型 ===")
base_scores = evaluate(base_model, test_data)
print("\n=== 微调模型 ===")
tuned_scores = evaluate(tuned_model, test_data)

# 结果统计
def print_stats(scores, name):
    valid_scores = [s for s in scores if s > 0]
    print(f"\n{name}统计:")
    print(f"有效样本: {len(valid_scores)}/{len(scores)}")
    print(f"平均ROUGE-L: {sum(valid_scores)/len(valid_scores):.3f}" if valid_scores else "无有效得分")

print_stats(base_scores, "原始模型")
print_stats(tuned_scores, "微调模型")