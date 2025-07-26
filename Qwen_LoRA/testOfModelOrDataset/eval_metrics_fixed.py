# eval_metrics_fixed.py
import json
import torch  # 添加此行
from rouge_score import rouge_scorer
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# 加载测试集
try:
    with open("tourism_test.jsonl") as f:
        test_data = [json.loads(line) for line in f]
except FileNotFoundError:
    print("❌ 请先运行 create_testset.py 生成测试文件")
    exit()

# 初始化评估工具
scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)

def evaluate(model, tokenizer):
    scores = []
    for item in test_data[:10]:  # 先测试前10条节省时间
        inputs = tokenizer(item["question"], return_tensors="pt").to("cuda")
        outputs = model.generate(**inputs, max_new_tokens=200)
        pred = tokenizer.decode(outputs[0], skip_special_tokens=True)
        scores.append(scorer.score(item["answer"], pred)["rougeL"].fmeasure)
    return sum(scores)/len(scores)

# 计算基线分数（原始模型）
base_model = AutoModelForCausalLM.from_pretrained(
    "/mnt/workspace/qwen_model/Qwen/Qwen2.5-7B-Instruct",
    device_map="auto",
    torch_dtype=torch.float16
)
tokenizer = AutoTokenizer.from_pretrained(
    "/mnt/workspace/qwen_model/Qwen/Qwen2.5-7B-Instruct",
    trust_remote_code=True
)
base_score = evaluate(base_model, tokenizer)

# 计算微调模型分数
tuned_model = PeftModel.from_pretrained(
    base_model,
    "/mnt/workspace/qwen_model/lora_results/final_adapter"
)
tuned_score = evaluate(tuned_model, tokenizer)

print(f"\n评估结果（Rouge-L分数）:")
print(f"原始模型: {base_score:.3f}")
print(f"微调模型: {tuned_score:.3f}")
print(f"提升比例: {(tuned_score-base_score)/base_score*100:.1f}%")