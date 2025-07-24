import numpy as np

# 定义判断矩阵
A = np.array([[1, 2, 3, 5], 
              [1/2, 1, 1/2, 2], 
              [1/3, 2, 1, 2], 
              [1/5, 1/2, 1/2, 1]])

n = A.shape[0]  # 获取矩阵阶数

# ======================
# 第一部分：一致性检验
# ======================
print("="*50)
print("判断矩阵一致性检验报告".center(40))
print("="*50)

# 计算特征值和特征向量
eig_val, eig_vec = np.linalg.eig(A)
max_eig = np.max(eig_val)  # 最大特征值[3](@ref)

# 计算一致性指标
CI = (max_eig - n) / (n-1)

# 随机一致性指标(RI值)[6](@ref)
RI = [0, 0, 0.52, 0.89, 1.12, 1.26, 1.36, 1.41, 1.46, 1.49, 1.52, 1.54, 1.56, 1.58, 1.59]
CR = CI / RI[n-1] if n > 1 else 0  # 处理1阶矩阵情况

# 格式化输出检验结果
print(f"矩阵阶数: {n}×{n}")
print(f"最大特征值: {max_eig:.6f}")
print(f"一致性指标 CI = {CI:.6f}")
print(f"随机一致性指标 RI = {RI[n-1]}")
print(f"一致性比例 CR = {CR:.6f}")

# 一致性判断
if CR < 0.10:
    print("\033[1;32m" + "✓ 一致性检验通过(CR < 0.10)" + "\033[0m")
else:
    print("\033[1;31m" + "✗ 一致性检验未通过(CR ≥ 0.10)，建议修改判断矩阵" + "\033[0m")

# ======================
# 第二部分：权重计算
# ======================
print("\n" + "="*50)
print("权重计算结果".center(40))
print("="*50)

# 获取最大特征值对应的特征向量
max_idx = np.argmax(eig_val)
max_vector = eig_vec[:, max_idx].real  # 取实部[3](@ref)

# 归一化得到权重
weights = max_vector / np.sum(max_vector)

# 格式化输出权重
print("各因素权重分配:")
for i, w in enumerate(weights):
    print(f"因素 {i+1}: {w:.6f} ({w*100:.2f}%)")

# 验证权重和
print(f"\n权重总和: {np.sum(weights):.10f}")

# ======================
# 第三部分：结果分析
# ======================
print("\n" + "="*50)
print("分析结论".center(40))
print("="*50)

# 确定主导因素
dominant_factor = np.argmax(weights) + 1
print(f"主导因素分析: 因素 {dominant_factor} 权重最高，是决策中最关键的因素")

# 权重分布评估
weight_diff = np.max(weights) - np.min(weights)
if weight_diff > 0.3:
    print("权重分布: 各因素权重差异显著，决策具有明显倾向性")
elif weight_diff > 0.15:
    print("权重分布: 各因素权重存在一定差异，决策需综合考虑")
else:
    print("权重分布: 各因素权重较为均衡，决策需平衡考虑")