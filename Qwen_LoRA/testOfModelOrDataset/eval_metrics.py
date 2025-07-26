# eval_metrics.py
from datasets import load_dataset
from rouge_score import rouge_scorer

# 加载测试集（可从tourism_data.jsonl抽取100条作为验证集）
test_data = load_dataset("json", data_files="tourism_test.jsonl")["train"]

scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)

def evaluate(model):
    scores = []
    for item in test_data:
        pred = generate_answer(model, item["question"])
        scores.append(scorer.score(item["answer"], pred)["rougeL"].fmeasure)
    return sum(scores)/len(scores)

base_score = evaluate(base_model)
tuned_score = evaluate(tuned_model)
print(f"原始模型Rouge-L: {base_score:.3f} | 微调模型Rouge-L: {tuned_score:.3f}")