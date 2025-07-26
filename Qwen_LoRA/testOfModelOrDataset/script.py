import json
import torch
import numpy as np
import pandas as pd
from rouge_score import rouge_scorer
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
from peft import PeftModel
from scipy import stats
import time
from tqdm import tqdm
import random
from collections import defaultdict
import argparse
import os

# ====================
# 测试集构建模块
# ====================

def extract_city(text, city_keywords):
    """从文本中提取城市名称"""
    for city in city_keywords:
        if city in text:
            return city
    return None

def build_test_set(input_file, output_file, city_keywords, test_ratio=0.2, min_samples_per_city=3):
    """
    从训练数据中构建测试集，确保城市多样性
    
    参数:
        input_file: 训练数据文件路径
        output_file: 测试数据输出路径
        city_keywords: 关注的城市列表
        test_ratio: 每个城市抽取的样本比例
        min_samples_per_city: 每个城市至少保留的样本数
    """
    print(f"从 {input_file} 构建测试集...")
    
    # 统计城市出现频率并收集有效样本
    city_counter = defaultdict(int)
    valid_pairs = []
    
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                data = json.loads(line)
                messages = data["messages"]
                
                # 验证数据格式
                if len(messages) >= 3 and \
                   messages[0]["role"] == "system" and \
                   messages[1]["role"] == "user" and \
                   messages[2]["role"] == "assistant":
                    
                    user_content = messages[1]["content"]
                    city = extract_city(user_content, city_keywords)
                    
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
            except Exception as e:
                print(f"解析数据行失败: {e}")
                continue
    
    print(f"原始数据中发现 {len(city_counter)} 个城市，{len(valid_pairs)} 条有效样本")
    
    # 按城市分层抽样
    test_data = []
    city_distribution = {}
    
    for city, count in city_counter.items():
        city_samples = [x for x in valid_pairs if x["city"] == city]
        
        # 确保每个城市至少有最小数量的样本
        test_size = max(min_samples_per_city, int(count * test_ratio))
        test_size = min(test_size, len(city_samples))
        
        if test_size > 0:
            selected_samples = random.sample(city_samples, test_size)
            test_data.extend(selected_samples)
            city_distribution[city] = test_size
    
    # 保存测试集
    with open(output_file, 'w', encoding='utf-8') as f:
        for item in test_data:
            f.write(json.dumps({
                "question": item["question"],
                "answer": item["answer"],
                "city": item["city"]  # 保留城市信息用于分析
            }, ensure_ascii=False) + "\n")
    
    print(f"测试集构建完成，共 {len(test_data)} 条数据")
    print("城市分布:")
    for city, count in sorted(city_distribution.items(), key=lambda x: -x[1]):
        print(f"  {city}: {count} 条")
    
    return test_data, city_distribution

# ====================
# 模型评估模块
# ====================

def load_model(model_path, lora_path=None):
    """加载基础模型和LoRA模型的替代方案"""
    print(f"加载模型: {model_path}")
    start_time = time.time()
    
    offload_dir = "./offload"
    os.makedirs(offload_dir, exist_ok=True)
    
    try:
        # 尝试使用更精细的设备映射
        base_model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map={
                "transformer.word_embeddings": 0,
                "transformer.word_embeddings_layernorm": 0,
                "transformer.h": {i: 0 for i in range(24)},  # 前24层放GPU
                "transformer.ln_f": "cpu",
                "lm_head": "cpu",
            },
            torch_dtype=torch.float16,
            trust_remote_code=True,
            offload_folder=offload_dir,
        ).eval()
        
        if lora_path:
            print(f"加载LoRA权重: {lora_path}")
            model = PeftModel.from_pretrained(
                base_model, 
                lora_path,
                device_map="auto",
                offload_dir=offload_dir,
            ).eval()
        else:
            model = base_model
            
    except Exception as e:
        print(f"使用精细设备映射失败: {e}")
        print("尝试使用更保守的加载方式...")
        
        # 更保守的加载方式
        base_model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map="auto",
            torch_dtype=torch.float16,
            trust_remote_code=True,
            offload_folder=offload_dir,
            low_cpu_mem_usage=True,
        ).eval()
        
        if lora_path:
            print(f"加载LoRA权重: {lora_path}")
            model = PeftModel.from_pretrained(
                base_model, 
                lora_path,
                device_map="auto",
                offload_dir=offload_dir,
                offload_state_dict=True,
            ).eval()
        else:
            model = base_model
    
    print(f"模型加载完成，耗时: {time.time()-start_time:.2f}秒")
    return model

def clean_text(text):
    """简单的文本清理"""
    return text.strip()

def get_scores(reference, prediction):
    """计算多种评估指标"""
    if not reference or not prediction:
        return {
            'rouge1': 0, 'rouge2': 0, 'rougeL': 0, 'rougeLsum': 0,
            'exact_match': 0, 'length_ratio': 0
        }
    
    # ROUGE分数
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL', 'rougeLsum'], use_stemmer=True)
    rouge_scores = scorer.score(reference, prediction)
    result = {rouge_type: rouge_scores[rouge_type].fmeasure for rouge_type in rouge_scores}
    
    # BERTScore (如果可用)
    result['bert_f1'] = 0  # 默认设为0，如果没有安装bert_score库
    
    # 简单准确率 (精确匹配)
    result['exact_match'] = 1.0 if reference == prediction else 0.0
    
    # 长度比率
    result['length_ratio'] = len(prediction) / len(reference) if len(reference) > 0 else 0
    
    return result

def evaluate(model, data, model_name="模型", tokenizer=None, generation_config=None):
    """评估模型并返回详细结果"""
    print(f"\n开始{model_name}评估，样本数: {len(data)}")
    results = []
    
    if not tokenizer:
        tokenizer = AutoTokenizer.from_pretrained("/mnt/workspace/qwen_model/Qwen/Qwen2.5-7B-Instruct", trust_remote_code=True)
    
    if not generation_config:
        generation_config = GenerationConfig(
            max_new_tokens=512,
            temperature=0.1,
            top_p=0.85,
            num_beams=1,
            early_stopping=True,
            no_repeat_ngram_size=3,
            do_sample=True
        )
    
    for i, item in enumerate(tqdm(data, desc=f"{model_name}评估进度")):
        try:
            question = clean_text(item["question"])
            reference = clean_text(item["answer"])
            city = item.get("city", "未知")
            
            # 生成回答
            inputs = tokenizer(question, return_tensors="pt").to(model.device)
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    generation_config=generation_config
                )
            prediction = tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # 移除问题部分（如果存在）
            if question in prediction:
                prediction = prediction.replace(question, "").strip()
            
            # 计算分数
            scores = get_scores(reference, prediction)
            
            # 记录结果
            results.append({
                "id": i,
                "city": city,
                "question": question,
                "reference": reference,
                "prediction": prediction,
                "scores": scores,
                "success": True
            })
            
            # 每20个样本打印一次示例
            if (i + 1) % 20 == 0:
                print(f"\n样本 {i+1} ({city}):")
                print(f"问题: {question[:80]}...")
                print(f"参考答案: {reference[:80]}...")
                print(f"模型回答: {prediction[:80]}...")
                print(f"ROUGE-L: {scores['rougeL']:.3f}")
                
        except Exception as e:
            print(f"\n处理样本 {i} ({city}) 失败: {str(e)}")
            results.append({
                "id": i,
                "city": city,
                "question": question,
                "reference": reference,
                "prediction": f"ERROR: {str(e)}",
                "scores": {k: 0 for k in ['rouge1', 'rouge2', 'rougeL', 'rougeLsum', 'bert_f1', 'exact_match', 'length_ratio']},
                "success": False
            })
    
    return results

def calculate_statistics(results, model_name="模型"):
    """计算评估结果的统计数据"""
    valid_results = [r for r in results if r["success"]]
    success_rate = len(valid_results) / len(results) if results else 0
    
    print(f"\n{model_name}评估统计:")
    print(f"总样本数: {len(results)}")
    print(f"成功样本数: {len(valid_results)} ({success_rate:.2%})")
    
    if not valid_results:
        return {}
    
    # 计算平均分数
    all_scores = [r["scores"] for r in valid_results]
    metrics = list(all_scores[0].keys())
    avg_scores = {metric: np.mean([s[metric] for s in all_scores]) for metric in metrics}
    
    # 按城市分析
    city_results = defaultdict(list)
    for result in valid_results:
        city_results[result["city"]].append(result)
    
    city_metrics = {}
    for city, city_data in city_results.items():
        city_scores = [r["scores"] for r in city_data]
        city_metrics[city] = {
            "samples": len(city_data),
            "avg_rougeL": np.mean([s["rougeL"] for s in city_scores])
        }
    
    # 打印主要指标
    print("\n主要评估指标:")
    for metric in ["rougeL", "rouge1", "rouge2", "exact_match"]:
        if metric in avg_scores:
            print(f"{metric.upper()}: {avg_scores[metric]:.4f}")
    
    # 打印城市分析结果
    print("\n按城市分析结果:")
    for city, metrics in sorted(city_metrics.items(), key=lambda x: -x[1]["avg_rougeL"]):
        print(f"  {city} ({metrics['samples']}样本): ROUGE-L={metrics['avg_rougeL']:.4f}")
    
    return {
        "success_rate": success_rate,
        "avg_scores": avg_scores,
        "city_metrics": city_metrics,
        "total_samples": len(results),
        "valid_samples": len(valid_results)
    }

def compare_models(base_results, tuned_results, metrics=None):
    """比较两个模型的性能差异，进行统计显著性检验"""
    if not metrics:
        metrics = ["rouge1", "rouge2", "rougeL", "rougeLsum"]
    
    print("\n=== 模型对比 ===")
    
    base_stats = calculate_statistics(base_results, "基础模型")
    tuned_stats = calculate_statistics(tuned_results, "微调模型")
    
    comparison = {}
    for metric in metrics:
        if metric not in base_stats["avg_scores"]:
            continue
            
        # 提取分数
        base_scores = [r["scores"][metric] for r in base_results if r["success"]]
        tuned_scores = [r["scores"][metric] for r in tuned_results if r["success"]]
        
        # 计算改进百分比
        improvement = (tuned_stats["avg_scores"][metric] - base_stats["avg_scores"][metric]) / base_stats["avg_scores"][metric] * 100
        
        # 进行配对t检验
        _, p_value = stats.ttest_rel(base_scores, tuned_scores)
        is_significant = p_value < 0.05
        
        comparison[metric] = {
            "base_score": base_stats["avg_scores"][metric],
            "tuned_score": tuned_stats["avg_scores"][metric],
            "improvement_percent": improvement,
            "p_value": p_value,
            "significant": is_significant
        }
        
        significance_marker = "**" if is_significant else ""
        print(f"{metric.upper()}: "
              f"基础模型={base_stats['avg_scores'][metric]:.4f}, "
              f"微调模型={tuned_stats['avg_scores'][metric]:.4f}, "
              f"改进={improvement:+.2f}% {significance_marker}")
    
    # 按城市比较
    print("\n=== 按城市比较模型性能 ===")
    cities = set(base_stats["city_metrics"].keys()).union(set(tuned_stats["city_metrics"].keys()))
    
    for city in cities:
        base_city = base_stats["city_metrics"].get(city, {"avg_rougeL": 0, "samples": 0})
        tuned_city = tuned_stats["city_metrics"].get(city, {"avg_rougeL": 0, "samples": 0})
        
        if base_city["samples"] > 0 and tuned_city["samples"] > 0:
            improvement = (tuned_city["avg_rougeL"] - base_city["avg_rougeL"]) / base_city["avg_rougeL"] * 100 if base_city["avg_rougeL"] > 0 else 0
            marker = "**" if improvement > 10 else ""
            print(f"{city} ({base_city['samples']}样本): "
                  f"基础模型ROUGE-L={base_city['avg_rougeL']:.4f}, "
                  f"微调模型ROUGE-L={tuned_city['avg_rougeL']:.4f}, "
                  f"改进={improvement:+.2f}% {marker}")
    
    return comparison

def save_results(base_results, tuned_results, comparison, city_distribution, output_file):
    """保存评估结果到JSON文件"""
    print(f"\n保存详细结果到: {output_file}")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump({
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "city_distribution": city_distribution,
            "base_results": base_results,
            "tuned_results": tuned_results,
            "comparison": comparison
        }, f, ensure_ascii=False, indent=2)

# ====================
# 主函数
# ====================

def main():
    parser = argparse.ArgumentParser(description='模型评估工具')
    parser.add_argument('--build_testset', action='store_true', help='构建测试集')
    parser.add_argument('--evaluate', action='store_true', help='评估模型')
    parser.add_argument('--input_file', type=str, default="/mnt/workspace/qwen_model/Qwen/Qwen2.5-7B-Instruct/tourism_data.jsonl", help='输入训练数据文件')
    parser.add_argument('--test_file', type=str, default="tourism_test_valid.jsonl", help='测试数据文件')
    parser.add_argument('--output_file', type=str, default="evaluation_results.json", help='输出结果文件')
    parser.add_argument('--model_path', type=str, default="/mnt/workspace/qwen_model/Qwen/Qwen2.5-7B-Instruct", help='基础模型路径')
    parser.add_argument('--lora_path', type=str, default="/mnt/workspace/qwen_model/round2_output/final", help='LoRA模型路径')
    args = parser.parse_args()
    
    # 扩展城市列表
    city_keywords = [
        "北京", "上海", "广州", "深圳", "成都", "杭州", "武汉", "西安", "南京", "长沙", 
        "昆明", "兰州", "厦门", "青岛", "重庆", "天津", "苏州", "宁波", "合肥", "福州", 
        "郑州", "济南", "哈尔滨", "长春", "沈阳", "石家庄", "太原", "西宁", "银川", "南宁",
        "贵阳", "乌鲁木齐", "拉萨", "呼和浩特", "大连", "珠海", "三亚", "丽江", "桂林"
    ]
    
    # 构建测试集
    if args.build_testset:
        test_data, city_distribution = build_test_set(
            input_file=args.input_file,
            output_file=args.test_file,
            city_keywords=city_keywords,
            test_ratio=0.2,
            min_samples_per_city=3
        )
    else:
        # 加载现有测试集
        print(f"加载现有测试集: {args.test_file}")
        try:
            with open(args.test_file, 'r', encoding='utf-8') as f:
                test_data = [json.loads(line) for line in f]
            
            # 统计城市分布
            city_distribution = defaultdict(int)
            for item in test_data:
                city_distribution[item.get("city", "未知")] += 1
            
            print(f"加载完成，共 {len(test_data)} 条数据")
            print("城市分布:")
            for city, count in sorted(city_distribution.items(), key=lambda x: -x[1]):
                print(f"  {city}: {count} 条")
                
        except Exception as e:
            print(f"加载测试集失败: {e}")
            print("请先使用 --build_testset 参数构建测试集")
            return
    
    # 评估模型
    if args.evaluate:
        # 加载模型
        base_model = load_model(args.model_path)
        tuned_model = load_model(args.model_path, args.lora_path)
        tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
        
        # 评估基础模型
        base_results = evaluate(base_model, test_data, "基础模型", tokenizer)
        
        # 评估微调模型
        tuned_results = evaluate(tuned_model, test_data, "微调模型", tokenizer)
        
        # 比较模型
        comparison = compare_models(base_results, tuned_results)
        
        # 保存结果
        save_results(base_results, tuned_results, comparison, city_distribution, args.output_file)
        
        print("\n评估完成!")
    else:
        print("请使用 --evaluate 参数进行模型评估")

if __name__ == "__main__":
    main()