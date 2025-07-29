# 导入所需库：
# numpy 用于数值计算，scipy.optimize.linprog 用于线性规划求解
# matplotlib.pyplot 用于绘图展示结果
import numpy as np
from scipy.optimize import linprog
import matplotlib.pyplot as plt
from matplotlib import rcParams

"""
==================== 多目标规划问题定义与基础设置 ====================
目标：通过不同权重组合，求解线性规划下的多目标优化，
      并分析权重对目标函数值、决策变量的敏感性
"""
# 定义线性规划的约束条件：A*x <= b（不等式约束）
A = [[-1, -1]]  # 约束 x[0] + x[1] >= 7
b = [-7]        
# 决策变量 x 的上下界
bounds = [(0, 5), (0, 6)]  # x1 ∈ [0,5], x2 ∈ [0,6]

"""
==================== 预定义权重组合的求解 ====================
对每个预定义的权重组合，构造目标函数并求解线性规划，输出结果
"""
weights = np.array([(0.4, 0.6), (0.5, 0.5), (0.3, 0.7)])  # 权重组合

print("=== 预定义权重组合的求解结果 ===")
for i, (w1, w2) in enumerate(weights):
    # 构造综合目标函数系数
    c = [w1 * 2 + w2 * 0.4, w1 * 5 + w2 * 0.3]
    
    # 求解线性规划
    result = linprog(c, A_ub=A, b_ub=b, bounds=bounds, method='highs')
    x_opt = result.x
    
    # 计算各指标
    f1 = 2 * x_opt[0] + 5 * x_opt[1]
    f2 = 0.4 * x_opt[0] + 0.3 * x_opt[1]
    
    # 输出结果
    print(f"组合{i+1}: w1={w1}, w2={w2}")
    print(f"  最优解: x1={x_opt[0]:.2f}, x2={x_opt[1]:.2f}")
    print(f"  目标值: f1={f1:.2f}, f2={f2:.2f}, 综合指标={result.fun:.2f}\n")

"""
==================== 敏感性分析 ====================
生成连续的权重区间（W1 从 0.1 到 0.5），
观察目标函数和决策变量如何随权重变化
"""
# 生成权重区间
n = 400  # 样本点数量
W1 = np.linspace(0.1, 0.5, n)  # w1均匀分布在[0.1,0.5]
W2 = 1 - W1  # w2 = 1 - w1

# 初始化存储数组
F1 = np.zeros(n)    # 目标函数f1的值
F2 = np.zeros(n)    # 目标函数f2的值
X1 = np.zeros(n)    # 决策变量x1的值
X2 = np.zeros(n)    # 决策变量x2的值
FVAL = np.zeros(n)  # 综合目标值

# 遍历权重区间进行敏感性分析
print("进行敏感性分析...")
for i in range(n):
    w1 = W1[i]
    w2 = W2[i]
    
    # 构造目标函数系数 - 修改为图片中的公式
    # 注意: 此处按照图片中的归一化处理
    c = [w1/30 * 2 + w2/2 * 0.4, 
         w1/30 * 5 + w2/2 * 0.3]
    
    # 求解线性规划
    result = linprog(c, A_ub=A, b_ub=b, bounds=bounds, method='highs')
    
    # 存储结果
    x = result.x
    F1[i] = 2 * x[0] + 5 * x[1]      # 目标f1值
    F2[i] = 0.4 * x[0] + 0.3 * x[1]  # 目标f2值
    X1[i] = x[0]                     # 决策变量x1
    X2[i] = x[1]                     # 决策变量x2
    FVAL[i] = result.fun             # 综合指标

print("敏感性分析完成!")

"""
==================== 绘图展示结果 ====================
"""
# 设置中文显示
rcParams['font.sans-serif'] = 'SimHei'
rcParams['axes.unicode_minus'] = False

# 创建1行3列的子图布局
plt.figure(figsize=(18, 5))

# 子图1: 目标函数值随权重变化
plt.subplot(131)
plt.plot(W1, F1, 'b-', label='f1')
plt.plot(W1, F2, 'r--', label='f2')
plt.xlabel("f1的权重(w1)")
plt.ylabel("目标函数值")
plt.title("目标函数值随权重变化的关系")
plt.legend()
plt.grid(True)

# 子图2: 决策变量随权重变化
plt.subplot(132)
plt.plot(W1, X1, 'g-', label='x1')
plt.plot(W1, X2, 'm--', label='x2')
plt.xlabel("f1的权重(w1)")
plt.ylabel("决策变量值")
plt.title("决策变量随权重变化的关系")
plt.legend()
plt.grid(True)

# 子图3: 综合指标随权重变化
plt.subplot(133)
plt.plot(W1, FVAL, 'k-')
plt.xlabel("f1的权重(w1)")
plt.ylabel("综合指标值")
plt.title("综合指标随权重变化的关系")
plt.grid(True)

plt.tight_layout()
plt.savefig('sensitivity_analysis.png', dpi=300)
plt.show()

print("图表已生成并保存为 sensitivity_analysis.png")