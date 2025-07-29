import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# ===== 解决中文显示问题 =====
plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体显示中文
plt.rcParams['axes.unicode_minus'] = False    # 解决负号显示问题

# ==================== 步骤1: 数据准备 ====================
# 根据文档表格创建数据集
data = {
    '需求': [100, 75, 80, 70, 50, 65, 90, 100, 110, 60],
    '收入': [1000, 600, 1200, 500, 300, 400, 1300, 1100, 1300, 300],
    '价格': [5, 7, 6, 6, 8, 7, 5, 4, 3, 9]
}
df = pd.DataFrame(data)

# 提取特征和标签
X = df[['收入', '价格']]
y = df['需求']

# ==================== 修正方法: 纯二次多项式回归 (仅平方项，无交叉项) ====================
# 创建仅包含原始变量和平方项的特征矩阵（排除交叉项）
# 手动添加平方项（与文档中的MATLAB方法一致）
X_manual = X.copy()
X_manual['收入²'] = X_manual['收入'] ** 2
X_manual['价格²'] = X_manual['价格'] ** 2

# 添加常数项（截距）
X_manual = np.c_[np.ones(len(X_manual)), X_manual]

# 创建并训练线性回归模型
model = LinearRegression(fit_intercept=False)  # 已手动添加常数项
model.fit(X_manual, y)

# 获取回归系数
coefficients = model.coef_
intercept = coefficients[0]  # 第一项为截距
a, b, c, d, e = coefficients  # 分别对应: 截距, 收入, 价格, 收入², 价格²
print("纯二次模型回归结果")
print(f"回归方程: y = {a:.4f} + {b:.4f}*收入 + {c:.4f}*价格 + {d:.4f}*收入² + {e:.4f}*价格²")

# 计算预测值和评估指标
y_pred = model.predict(X_manual)
rmse = np.sqrt(mean_squared_error(y, y_pred))
r2 = r2_score(y, y_pred)
print(f"剩余标准差(RMSE): {rmse:.4f}")
print(f"R²决定系数: {r2:.4f}")

# ==================== 步骤2: 预测特定值 ====================
# 预测收入=1000, 价格=6时的需求量
new_data = np.array([[1, 1000, 6, 1000**2, 6**2]])  # 手动创建特征数组[常数项, 收入, 价格, 收入², 价格²]
prediction = model.predict(new_data)[0]
print(f"\n预测结果: 当收入=1000, 价格=6时, 预测需求量为: {prediction:.4f}")

# ==================== 步骤3: 可视化分析 ====================
# 创建3D图形展示数据点和回归曲面
fig = plt.figure(figsize=(14, 10))
ax = fig.add_subplot(111, projection='3d')

# 绘制原始数据点
ax.scatter(df['收入'], df['价格'], df['需求'], 
           c='red', s=100, marker='o', label='原始数据点')

# 创建网格数据用于绘制回归曲面
x1_range = np.linspace(df['收入'].min(), df['收入'].max(), 20)
x2_range = np.linspace(df['价格'].min(), df['价格'].max(), 20)
X1, X2 = np.meshgrid(x1_range, x2_range)

# 准备预测数据（手动创建特征）
grid = np.c_[np.ones(X1.size), X1.ravel(), X2.ravel(), X1.ravel()**2, X2.ravel()**2]
Z = model.predict(grid).reshape(X1.shape)

# 绘制回归曲面
surf = ax.plot_surface(X1, X2, Z, rstride=1, cstride=1,
                       alpha=0.3, color='blue', label='回归曲面')

# 标记预测点
ax.scatter([1000], [6], [prediction], 
           c='green', s=200, marker='*', label='预测点(收入=1000, 价格=6)')

# 设置图形属性
ax.set_xlabel('收入', fontsize=12, labelpad=10)
ax.set_ylabel('价格', fontsize=12, labelpad=10)
ax.set_zlabel('需求', fontsize=12, labelpad=10)
ax.set_title('商品需求回归模型(纯二次模型)', fontsize=16, pad=20)
ax.legend(loc='upper left', fontsize=10)

# 添加回归方程注释
equation_text = (f"回归方程: $y = {a:.4f} + {b:.4f}x_1 + {c:.4f}x_2 + {d:.7f}x_1^2 + {e:.4f}x_2^2$\n"
                 f"$R^2 = {r2:.4f}$, RMSE = {rmse:.4f}\n"
                 f"预测值(1000,6): {prediction:.4f} (Matlab结果: 88.47981)")
plt.figtext(0.5, 0.01, equation_text, ha='center', fontsize=12, 
            bbox={"facecolor":"orange", "alpha":0.2, "pad":5})

plt.tight_layout()
plt.show()