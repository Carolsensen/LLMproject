# fixed_cleaner.py
import json

def clean_question(q):
    """处理问题中的固定模板"""
    templates = [
        "你是一名专业的旅游助手，熟悉中国各地旅游景点和行程规划",
        "你是一名专业的旅游助手"
    ]
    for tpl in templates:
        q = q.replace(tpl, "").strip(" ，。")
    return q if q else None

# 新清洗逻辑
with open("tourism_test_fixed.jsonl") as f_in, open("fixed_test.jsonl", "w") as f_out:
    for line in f_in:
        try:
            data = json.loads(line)
            q = clean_question(data["question"])
            a = data["answer"].strip()
            
            # 新过滤条件（更宽松）
            if q and a and len(q) > 3 and len(a) > 2:
                f_out.write(json.dumps({
                    "question": q,
                    "answer": a
                }, ensure_ascii=False) + "\n")
        except:
            continue

print("清洗完成，请检查 fixed_test.jsonl")