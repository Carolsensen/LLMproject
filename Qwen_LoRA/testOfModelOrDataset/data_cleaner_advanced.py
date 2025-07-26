# data_cleaner_advanced.py
# data_cleaner_advanced_fixed.py
import json
import re
from collections import defaultdict

def extract_city(text):
    """城市提取函数（完整版）"""
    city_keywords = [
        "北京", "上海", "广州", "深圳", "成都", "杭州", "武汉", "西安", "南京", 
        "长沙", "昆明", "兰州", "厦门", "青岛", "苏州", "桂林", "三亚", "敦煌"
    ]
    for city in city_keywords:
        if city in text:
            return city
    return None

def is_valid_pair(q, a):
    """六重过滤标准（完整实现）"""
    # 1. 问答长度检查
    if len(q) < 6 or len(a) < 15:
        return False
    
    # 2. 问题有效性
    if not any(w in q for w in ["吗","？","怎么","如何","什么","哪","推荐"]) and "?" not in q:
        return False
        
    # 3. 答案完整性
    if a.count('，') < 2 and a.count('。') < 1:
        return False
    
    # 4. 特殊符号过滤
    if re.search(r'[<>{}【】]', q + a):
        return False
        
    # 5. 重复内容检查
    if q in a or a in q:
        return False
        
    # 6. 城市均衡性
    city = extract_city(q)
    return city is not None

# 主处理逻辑
input_file = "tourism_test_valid.jsonl"
output_file = "cleaned_test_v2.jsonl"

city_counter = defaultdict(int)
with open(input_file) as f, open(output_file, "w") as out:
    for line in f:
        try:
            data = json.loads(line)
            q = data["question"]
            a = data["answer"]
            
            if is_valid_pair(q, a):
                city = extract_city(q)
                city_counter[city] += 1
                
                # 标准化输出
                out.write(json.dumps({
                    "question": q,
                    "answer": a[:300]  # 限制长度
                }, ensure_ascii=False) + "\n")
        except json.JSONDecodeError:
            continue

print(f"清洗完成！有效数据分布：{dict(city_counter)}")
print(f"输出文件：{output_file}")