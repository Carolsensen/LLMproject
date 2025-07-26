# check_testdata.py
import json

with open("tourism_test.jsonl") as f:
    sample = json.loads(next(f))
    
print("问题类型:", type(sample["question"]), "示例:", sample["question"][:50])
print("答案类型:", type(sample["answer"]), "示例:", sample["answer"][:50])