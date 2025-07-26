# recreate_testset.py
import json
with open("/mnt/workspace/qwen_model/Qwen/Qwen2.5-7B-Instruct/tourism_data.jsonl") as f:
    with open("tourism_test_fixed.jsonl", "w") as out:
        for line in f:
            data = json.loads(line)
            out.write(json.dumps({
                "question": data["messages"][0]["content"],
                "answer": data["messages"][1]["content"]
            }) + "\n")