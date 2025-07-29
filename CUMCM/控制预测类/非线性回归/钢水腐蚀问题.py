import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# ===== 解决中文显示问题 =====
plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体显示中文
plt.rcParams['axes.unicode_minus'] = False    # 解决负号显示问题

# ==================== 步骤1: 定义非线性函数模型 ====================
# 根据文档中的倒指数曲线模型: y = a * exp(b/x)
def exp_model(x, a, b):
    """倒指数曲线模型"""
    return a * np.exp(b / x)

# ==================== 步骤2: 准备数据 ====================
# 文档中的钢包侵蚀数据 (使用次数从2开始，因x=1时数据缺失)
x_data = np.array([2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16])
y_data = np.array([6.42, 8.20, 9.58, 9.50, 9.70, 10.00, 9.93, 9.99, 
                   10.49, 10.59, 10.60, 10.80, 10.60, 10.90, 10.76])

# ==================== 步骤3: 绘制原始散点图 ====================
# 文档要求首先画出散点图观察数据分布
plt.figure(figsize=(10, 6))
plt.scatter(x_data, y_data, c='k', marker='+', s=100, label='原始数据')
plt.xlabel('使用次数')
plt.ylabel('增大容积')
plt.title('钢包使用次数与容积增大关系散点图')
plt.grid(True, linestyle='--', alpha=0.7)

# ==================== 步骤4: 非线性回归拟合 ====================
# 执行曲线拟合（参考文档中的初始参数 beta0=[8, 2]）
p0 = [8, 2]  # 初始参数估计值
params, cov = curve_fit(exp_model, x_data, y_data, p0=p0)

# 解析拟合参数
a_fit, b_fit = params
print(f"拟合参数: a = {a_fit:.4f}, b = {b_fit:.4f}")

# ==================== 步骤5: 生成拟合曲线 ====================
x_fit = np.linspace(2, 16, 100)
y_fit = exp_model(x_fit, a_fit, b_fit)

# 绘制拟合曲线（红色曲线，参考文档中的可视化要求）
plt.plot(x_fit, y_fit, 'r-', linewidth=2, label=f'拟合曲线: $y = {a_fit:.2f}e^{{{b_fit:.2f}/x}}$')

# ==================== 步骤6: 模型评估 ====================
# 计算预测值和残差
y_pred = exp_model(x_data, a_fit, b_fit)
residuals = y_data - y_pred

# 计算R²决定系数
ss_res = np.sum(residuals**2)
ss_tot = np.sum((y_data - np.mean(y_data))**2)
r_squared = 1 - (ss_res / ss_tot)
print(f"模型R²值: {r_squared:.4f}")

# ==================== 步骤7: 可视化结果 ====================
plt.legend(fontsize=12)
plt.annotate(f'$R^2 = {r_squared:.3f}$', xy=(0.7, 0.15), xycoords='axes fraction', fontsize=14)

# 添加残差图
plt.figure(figsize=(10, 4))
plt.scatter(x_data, residuals, c='b', marker='o', s=80)
plt.axhline(y=0, color='r', linestyle='--')
plt.xlabel('使用次数')
plt.ylabel('残差')
plt.title('模型残差分析')
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()

plt.show()