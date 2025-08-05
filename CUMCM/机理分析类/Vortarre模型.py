import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

# 设置中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']  
plt.rcParams['axes.unicode_minus'] = False

# 定义Lotka-Volterra模型的微分方程
def volterra_model(X, t, alpha, beta, delta, gamma):
    prey, predator = X  # prey: 猎物数量, predator: 捕食者数量
    d_prey_dt = alpha * prey - beta * prey * predator  # 猎物增长率
    d_predator_dt = delta * prey * predator - gamma * predator  # 捕食者增长率
    return [d_prey_dt, d_predator_dt]

# 模型参数设置
alpha = 0.6   # 猎物自然增长率
beta = 0.02   # 捕食率系数
delta = 0.01  # 捕食者增长率系数
gamma = 0.5   # 捕食者自然死亡率

# 初始条件
initial_prey = 50    # 初始猎物数量
initial_predator = 20 # 初始捕食者数量
X0 = [initial_prey, initial_predator]

# 时间范围
t = np.linspace(0, 100, 1000)  # 0到100时间单位，1000个采样点

# 求解微分方程
solution = odeint(volterra_model, X0, t, args=(alpha, beta, delta, gamma))
prey_population = solution[:, 0]
predator_population = solution[:, 1]

# 创建图形
fig = plt.figure(figsize=(12, 10))
gs = GridSpec(2, 1, height_ratios=[1, 1])

# 第一个子图：种群随时间变化
ax1 = fig.add_subplot(gs[0, 0])
ax1.plot(t, prey_population, label='猎物', color='blue', linewidth=2)
ax1.plot(t, predator_population, label='捕食者', color='red', linewidth=2)
ax1.set_xlabel('时间', fontsize=12)
ax1.set_ylabel('种群数量', fontsize=12)
ax1.set_title('Lotka-Volterra模型：种群数量随时间变化', fontsize=14)
ax1.legend(fontsize=12)
ax1.grid(True, linestyle='--', alpha=0.7)

# 第二个子图：相轨线图（捕食者数量 vs 猎物数量）
ax2 = fig.add_subplot(gs[1, 0])
ax2.plot(prey_population, predator_population, color='green', linewidth=2)

# 标记初始点
ax2.scatter(initial_prey, initial_predator, color='purple', s=100, zorder=5, 
            label=f'初始点 ({initial_prey}, {initial_predator})')

ax2.set_xlabel('猎物数量', fontsize=12)
ax2.set_ylabel('捕食者数量', fontsize=12)
ax2.set_title('相轨线图：捕食者数量 vs 猎物数量', fontsize=14)
ax2.legend(fontsize=12)
ax2.grid(True, linestyle='--', alpha=0.7)

# 调整布局
plt.tight_layout()
plt.show()
