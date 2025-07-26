# diagnose_data.py
import json

with open("tourism_test_fixed.jsonl") as f:
    first_line = json.loads(next(f))
    
print("=== 数据结构分析 ===")
print("完整样本:", first_line)
print("\n问题类型:", type(first_line["question"]), "长度:", len(first_line["question"]))
print("答案类型:", type(first_line["answer"]), "长度:", len(first_line["answer"]))
print("\n问题示例:", first_line["question"][:100])
print("答案示例:", first_line["answer"][:100])