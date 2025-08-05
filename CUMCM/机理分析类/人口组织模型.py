import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
import seaborn as sns

# ===== 解决中文显示问题 =====
plt.rcParams['font.sans-serif'] = ['SimHei']  
plt.rcParams['axes.unicode_minus'] = False

# 1. 定义阻滞增长模型的微分方程
def logistic_growth_model(P, t, r, K):
    dPdt = r * P * (1 - P / K)
    return dPdt

# 2. 定义初始条件和参数
P0 = 100       # 初始人口数量
t = np.linspace(0, 1000, 1000)  # 时间范围（0 到 1000，共 1000 个点）
r = 0.04       # 人口增长率
K = 1000       # 环境容量

# 3. 求解微分方程
P = odeint(logistic_growth_model, P0, t, args=(r, K))

# 4. 使用 seaborn 美化绘图
sns.set_style("whitegrid")  # 设置白色网格风格
sns.set_palette("colorblind")  # 设置适合色盲人群的调色板
sns.set_context("talk", font_scale=1.2)  # 设置绘图上下文（字体更大，适合展示）

# 5. 绘制人口随时间变化的图像
plt.figure(figsize=(10, 6))  # 设置图像大小
plt.plot(t, P, linewidth=2, color=sns.color_palette()[0])  # 绘制曲线，线条更粗
plt.xlabel('Time', fontsize=14)  # 设置 x 轴标签
plt.ylabel('Population', fontsize=14)  # 设置 y 轴标签
plt.title('Logistic Growth Model', fontsize=16, pad=20)  # 设置标题，增加间距
plt.grid(True, linestyle='--', alpha=0.7)  # 优化网格样式
sns.despine(top=True, right=True)  # 移除顶部和右侧边框
plt.tight_layout()  # 自动调整布局，避免标签截断
plt.show()  # 显示图像