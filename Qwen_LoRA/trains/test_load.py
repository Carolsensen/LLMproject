# test_load.py
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# 配置参数
MODEL_PATH = "/mnt/workspace/qwen_model/Qwen/Qwen2.5-7B-Instruct"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

try:
    # 1. 加载tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_PATH,
        trust_remote_code=True
    )
    print("✅ Tokenizer加载成功")

    # 2. 加载模型（4-bit量化节省显存）
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        device_map="auto",
        torch_dtype=torch.float16,
        load_in_4bit=True,  # 24GB显存建议启用
        trust_remote_code=True
    )
    print(f"✅ 模型加载成功 | 显存占用: {torch.cuda.memory_allocated()/1024**3:.1f}GB")

    # 3. 测试推理
    input_text = "天水麦积山石窟有什么特色？"
    inputs = tokenizer(input_text, return_tensors="pt").to(DEVICE)
    
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=100)
    
    print("\n🔍 测试输出：")
    print(tokenizer.decode(outputs[0], skip_special_tokens=True))

except Exception as e:
    print(f"❌ 加载失败: {str(e)}")
    print("\n排查建议：")
    print("1. 检查文件权限：ls -l /mnt/workspace/qwen_model/")
    print("2. 验证CUDA是否可用：python -c 'import torch; print(torch.cuda.is_available())'")
    print("3. 尝试简化加载：先仅加载tokenizer测试")