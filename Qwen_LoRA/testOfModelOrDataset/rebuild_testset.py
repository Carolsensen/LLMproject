# rebuild_testset.py
import json
import random
from collections import defaultdict

def extract_valid_pairs(input_file, output_file, test_ratio=0.2):
    """从训练数据中提取测试集"""
    # 统计城市出现频率
    city_counter = defaultdict(int)
    valid_pairs = []
    
    with open(input_file) as f:
        for line in f:
            try:
                data = json.loads(line)
                messages = data["messages"]
                
                # 验证数据格式
                if len(messages) >= 3 and \
                   messages[0]["role"] == "system" and \
                   messages[1]["role"] == "user" and \
                   messages[2]["role"] == "assistant":
                    
                    # 提取城市关键词（示例：从"请问兰州有哪些景点？"提取"兰州"）
                    user_content = messages[1]["content"]
                    city = extract_city(user_content)  # 需要自定义城市提取函数
                    if city:
                        city_counter[city] += 1
                        
                        # 处理助理回复（移除/no_think标记）
                        assistant_content = messages[2]["content"]
                        if assistant_content.startswith("/no_think"):
                            assistant_content = assistant_content[9:].strip()
                        
                        valid_pairs.append({
                            "city": city,
                            "question": user_content,
                            "answer": assistant_content
                        })
            except:
                continue
    
    # 按城市分层抽样
    test_data = []
    for city, count in city_counter.items():
        city_samples = [x for x in valid_pairs if x["city"] == city]
        test_size = max(1, int(count * test_ratio))
        test_data.extend(random.sample(city_samples, min(test_size, len(city_samples))))
    
    # 保存测试集
    with open(output_file, "w") as f:
        for item in test_data:
            f.write(json.dumps({
                "question": item["question"],
                "answer": item["answer"]
            }, ensure_ascii=False) + "\n")
    
    print(f"生成测试集完成，总计{len(test_data)}条数据")
    print("城市分布：", dict(sorted(city_counter.items(), key=lambda x: -x[1])[:5]))

def extract_city(text):
    """简易城市提取函数（需根据实际数据调整）"""
    city_keywords = ["北京", "上海", "广州", "深圳", "成都", "杭州", "武汉", 
                    "西安", "南京", "长沙", "昆明", "兰州", "厦门", "青岛"]
    for city in city_keywords:
        if city in text:
            return city
    return None

# 使用示例
extract_valid_pairs(
    input_file="/mnt/workspace/qwen_model/Qwen/Qwen2.5-7B-Instruct/tourism_data.jsonl",
    output_file="tourism_test_valid.jsonl"
)