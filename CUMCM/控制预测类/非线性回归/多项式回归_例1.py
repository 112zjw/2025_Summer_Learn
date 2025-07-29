import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# ===== 解决中文显示问题 =====
plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体显示中文
plt.rcParams['axes.unicode_minus'] = False    # 解决负号显示问题

# ==================== 步骤1: 数据准备 ====================
# 根据文档表格中的数据
# t = [1/30, 2/30, ..., 14/30]
t_values = np.array([1/30, 2/30, 3/30, 4/30, 5/30, 6/30, 7/30, 
                     8/30, 9/30, 10/30, 11/30, 12/30, 13/30, 14/30])
s_values = np.array([11.86, 15.67, 20.60, 26.69, 33.71, 41.93, 51.13,
                     61.49, 72.90, 85.44, 99.08, 113.77, 129.54, 146.48])

# ==================== 步骤2: 可视化原始数据 ====================
plt.figure(figsize=(12, 8))
plt.scatter(t_values, s_values, c='k', marker='+', s=100, label='原始数据')
plt.title('物体降落距离与时间关系散点图', fontsize=16)
plt.xlabel('时间 (s)', fontsize=14)
plt.ylabel('距离 (cm)', fontsize=14)
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend(fontsize=12)

# ==================== 方法1: 直接多项式拟合 ====================
# 使用numpy的polyfit函数进行二次多项式拟合
# 与文档中MATLAB的polyfit函数等效
coeffs = np.polyfit(t_values, s_values, 2)
a_fit, b_fit, c_fit = coeffs
print("多项式拟合结果:")
print(f"回归方程: s = {a_fit:.4f}t² + {b_fit:.4f}t + {c_fit:.4f}")

# 生成拟合曲线数据点
t_fit = np.linspace(min(t_values), max(t_values), 100)
s_fit = np.polyval(coeffs, t_fit)

# 计算R²值
s_pred = np.polyval(coeffs, t_values)
r2 = r2_score(s_values, s_pred)
print(f"R²决定系数: {r2:.6f}")

# 可视化拟合结果
plt.plot(t_fit, s_fit, 'r-', linewidth=2, 
         label=f'多项式拟合: $s = {a_fit:.2f}t^2 + {b_fit:.2f}t + {c_fit:.2f}$')

# ==================== 方法2: 多元线性回归 ====================
# 创建多项式特征矩阵 (包含常数项、t和t²)
# 与文档中MATLAB的T = [ones(14,1), t', (t.^2)']等效
poly = PolynomialFeatures(degree=2, include_bias=True)
X_poly = poly.fit_transform(t_values.reshape(-1, 1))

# 创建并训练线性回归模型
model = LinearRegression()
model.fit(X_poly, s_values)

# 解析回归系数
coeffs_linear = model.coef_
intercept = model.intercept_
print("\n多元线性回归结果:")
print(f"回归方程: s = {coeffs_linear[2]:.4f}t² + {coeffs_linear[1]:.4f}t + {intercept:.4f}")

# 计算R²值
r2_linear = model.score(X_poly, s_values)
print(f"R²决定系数: {r2_linear:.6f}")

# ==================== 结果可视化与比较 ====================
# 添加图例和统计信息
plt.legend(fontsize=12)
plt.annotate(f'$R^2 = {r2:.5f}$', xy=(0.7, 0.1), xycoords='axes fraction', 
             fontsize=14, bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))

# 添加残差分析图
plt.figure(figsize=(12, 5))
residuals = s_values - s_pred
plt.scatter(t_values, residuals, c='b', marker='o', s=80)
plt.axhline(y=0, color='r', linestyle='--', linewidth=1.5)
plt.title('回归模型残差分析', fontsize=16)
plt.xlabel('时间 (s)', fontsize=14)
plt.ylabel('残差', fontsize=14)
plt.grid(True, linestyle='--', alpha=0.6)

plt.tight_layout()
plt.show()