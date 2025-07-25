import pandas as pd
import numpy as np
from scipy import linalg

# ======================
# 步骤1: 数据读取与准备 (与文件代码一致)
# ======================
try:
    # 读取Excel文件中的C到G列数据（对应5个变量）
    df = pd.read_excel(r'D:\2025_Summer_Learn\CUMCM\评价决策类\棉花产量论文作业的数据.xlsx', usecols='C:G')
except ImportError as e:
    print(f"导入错误: {e}")
    print("请安装openpyxl库: pip install openpyxl")
    exit(1)

print("原始数据预览:")
print(df)  # 只显示前5行避免输出过长
print("\n数据形状(行数, 列数):", df.shape)

# 将DataFrame转换为NumPy数组以便后续计算
data_matrix = df.to_numpy()
print("\n转换后的NumPy数组:")
print(data_matrix[:])  

# ======================
# 步骤2: 数据标准化 (与文件代码一致)
# ======================
# Z-score标准化: (原始值 - 均值)/标准差
# ddof=1 表示使用样本标准差（分母为n-1）
data_mean = np.mean(data_matrix, axis=0)  # 计算每列的均值
data_std = np.std(data_matrix, ddof=1, axis=0)  # 计算每列的标准差
X_standardized = (data_matrix - data_mean) / data_std

print("\n标准化后的数据:")
print(X_standardized[:5])  # 显示前5行
print("\n标准化后数据的均值:", np.round(np.mean(X_standardized, axis=0), 3))
print("标准化后数据的标准差:", np.round(np.std(X_standardized, ddof=1, axis=0), 3))

# ======================
# 步骤3: 计算协方差矩阵 (与文件代码一致)
# ======================
R = np.cov(X_standardized.T)  # T表示转置，计算列之间的协方差

print("\n协方差矩阵:")
print(np.round(R, 3))  # 保留3位小数便于阅读

# ======================
# 步骤4: 特征分解 (使用文件代码的排序方法)
# ======================
# 计算协方差矩阵的特征值和特征向量
eigenvalues, eigenvectors = linalg.eigh(R)

# 对特征值和特征向量进行降序排序
# (linalg.eigh默认返回升序排列)
sorted_indices = np.argsort(eigenvalues)[::-1]  # 获取降序排列的索引
eigenvalues = eigenvalues[sorted_indices]
eigenvectors = eigenvectors[:, sorted_indices]

print("\n特征值(降序排列):")
print(np.round(eigenvalues, 4))
print("\n对应的特征向量矩阵(每列为一个特征向量):")
print(np.round(eigenvectors, 4))

# ======================
# 步骤5: 计算主成分贡献率 (使用文件代码的表格格式)
# ======================
# 贡献率 = 单个特征值 / 所有特征值之和
# 累积贡献率 = 前k个主成分贡献率之和
total_variance = np.sum(eigenvalues)
contribution_rate = eigenvalues / total_variance
cumulative_contribution = np.cumsum(contribution_rate)

# 创建贡献率表格 (与文件代码一致)
contribution_table = pd.DataFrame({
    '特征值': eigenvalues,
    '贡献率(%)': contribution_rate * 100,
    '累积贡献率(%)': cumulative_contribution * 100
})
contribution_table.index = [f'主成分{i+1}' for i in range(len(eigenvalues))]

print("\n主成分贡献率分析:")
print(contribution_table.round(3))

# ======================
# 步骤6: 结果解释 (添加文件代码的结果解释部分)
# ======================
# 根据累积贡献率确定保留的主成分数量
# 通常选择累积贡献率>85%的主成分
recommended_components = np.argmax(cumulative_contribution >= 0.85) + 1
if recommended_components == 0:  # 处理特殊情况
    recommended_components = len(eigenvalues)
    
print(f"\n建议保留的主成分数量: {recommended_components} (累积贡献率 > 85%)")

# 输出主成分表达式 (文件代码的表达式格式)
print("\n主成分表达式(前3个):")
for i in range(min(3, len(eigenvalues))):  # 安全处理，避免索引越界
    expr = " + ".join([f"({coef:.3f}*X{j+1})" for j, coef in enumerate(eigenvectors[:, i])])
    print(f"PC{i+1} = {expr}")