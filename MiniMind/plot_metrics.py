import os
import matplotlib.pyplot as plt
import pandas as pd

# 手动指定CSV文件路径（与你的CSV位置一致）
metrics_path = os.path.join("outputs", "pretrain_a10", "metrics.csv")
# 图片保存路径（与CSV同目录）
save_path = os.path.join("outputs", "pretrain_a10", "metrics.png")

# 读取指标文件
df = pd.read_csv(metrics_path)

# 绘制损失曲线和学习率曲线
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(df['step'], df['loss'], label='Training Loss')
plt.xlabel('Step')
plt.ylabel('Loss')
plt.title('Training Loss Curve')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(df['step'], df['lr'], label='Learning Rate', color='orange')
plt.xlabel('Step')
plt.ylabel('LR')
plt.title('Learning Rate Schedule')
plt.legend()

# 保存图表
plt.tight_layout()
plt.savefig(save_path)
plt.close()

print(f"图表已保存至：{save_path}")