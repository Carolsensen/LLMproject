# validate_testset.py
import json

def validate(file_path):
    with open(file_path) as f:
        for i, line in enumerate(f, 1):
            data = json.loads(line)
            assert "question" in data, f"第{i}行缺少question字段"
            assert "answer" in data, f"第{i}行缺少answer字段"
            assert len(data["question"]) > 5, f"第{i}行问题过短"
            assert len(data["answer"]) > 10, f"第{i}行答案过短"
            assert not data["question"].startswith("你是一名"), "问题包含系统提示"
    print("测试集验证通过！")

validate("tourism_test_valid.jsonl")