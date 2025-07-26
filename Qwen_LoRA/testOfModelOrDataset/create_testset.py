# create_testset.py
import json
from random import sample

# 从原始训练数据中抽取100条作为测试集
with open("/mnt/workspace/qwen_model/Qwen/Qwen2.5-7B-Instruct/tourism_data.jsonl") as f:
    all_data = [json.loads(line) for line in f]

test_data = sample(all_data, min(100, len(all_data)))

# 保存测试集
with open("tourism_test.jsonl", "w") as f:
    for item in test_data:
        f.write(json.dumps({"question": item["messages"][0]["content"], 
                          "answer": item["messages"][1]["content"]}) + "\n")

print(f"已创建测试集：tourism_test.jsonl（{len(test_data)}条）")