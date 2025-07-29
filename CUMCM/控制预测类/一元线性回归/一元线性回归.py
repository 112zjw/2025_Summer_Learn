import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy import stats

# ===== 解决中文显示问题 =====
plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体显示中文
plt.rcParams['axes.unicode_minus'] = False    # 解决负号显示问题

# ===== 1. 数据准备 =====
"""
步骤1：准备数据
• x: 身高数据（单位：厘米）
• Y: 体重数据（单位：公斤）
• 数据来源：16个样本的观测值
"""
x = np.array([143, 145, 146, 147, 149, 150, 153, 154, 155, 156, 157, 158, 159, 160, 162, 164])
Y = np.array([88, 85, 88, 91, 92, 93, 93, 95, 96, 98, 97, 96, 98, 99, 100, 102])

# 添加常数项（截距）
X = sm.add_constant(x)  # 等价于 [ones(16,1), x]

# ===== 2. 回归分析 =====
"""
步骤2：进行普通最小二乘（OLS）回归分析
目标：建立体重(Y)与身高(X)之间的线性关系
模型公式：Y = β₀ + β₁*X + ε
"""
model = sm.OLS(Y, X).fit()

# 获取关键统计量
b = model.params          # 回归系数 [β₀, β₁]
bint = model.conf_int()   # 95%置信区间
r_squared = model.rsquared  # R²值
mse_resid = model.mse_resid  # 残差方差
dof_resid = model.df_resid  # 残差自由度

# ===== 新增：估计标准误差（SEE）计算 =====
"""
估计标准误差（Standard Error of Estimate, SEE）：
衡量观测值围绕回归线的离散程度，计算公式：
SEE = √(Σ(yᵢ - ŷᵢ)²/(n-2)) = √(MSE)
其中：
• n为样本量
• MSE为均方误差
"""
see = np.sqrt(mse_resid)  # 估计标准误差

# ===== 输出结果 =====
print("===== 回归分析结果 =====")
print(f"回归系数: β₀ = {b[0]:.3f}, β₁ = {b[1]:.3f}")
print(f"置信区间: β₀ ∈ [{bint[0][0]:.3f}, {bint[0][1]:.3f}]")
print(f"         β₁ ∈ [{bint[1][0]:.3f}, {bint[1][1]:.3f}]")
print(f"\n===== 统计指标 =====")
print(f"R² (拟合优度): {r_squared:.4f}")
print(f"MSE (均方误差): {mse_resid:.3f}")
print(f"SEE (估计标准误差): {see:.3f}")

# ===== 统计意义解释 =====
print("\n===== 统计意义解释 =====")
print("1. R² (拟合优度):")
print("   - 取值范围[0,1]，表示模型解释的变异比例")
print(f"   - {r_squared*100:.1f}%的腿长变异可由身高解释")
print("   - 值越接近1，模型拟合越好")

print("\n2. SEE (估计标准误差):")
print("   - 表示观测值围绕回归线的典型距离")
print("   - 单位与因变量相同（此处为腿长厘米）")
print(f"   - 预测值与实际值的平均差异约为{see:.1f}厘米")
print("   - 值越小，模型预测越精确")

# ===== 残差分析（先计算统计量，后绘图）=====
residuals = model.resid  # 获取残差
influence = model.get_influence()
summary_df = influence.summary_frame()

# 标准化残差（更稳健的异常值检测）
if 'standard_resid' in summary_df.columns:
    std_residuals = summary_df['standard_resid']
else:
    std_residuals = (residuals - residuals.mean()) / np.sqrt(mse_resid)

# 计算异常点（先输出，后绘图）
outliers = np.where(np.abs(std_residuals) > 2)[0] + 1
print(f"异常点序号: {outliers}")

# ===== 预测170cm身高的腿长 =====
x0 = 170
X0 = [1, x0]  # 添加常数项

# 点预测
y0_pred = model.predict(X0)[0]

# 预测区间
prediction = model.get_prediction(X0)
pred_interval = prediction.conf_int(alpha=0.05)[0]  # 95%预测区间

print("\n===== 身高170cm预测结果 =====")
print(f"预测腿长: {y0_pred:.3f} cm")
print(f"95%预测区间: [{pred_interval[0]:.3f}, {pred_interval[1]:.3f}] cm")

# ===== 模型检验结果 =====
print("\n===== 模型检验结果 =====")
# F检验部分
print("1. 模型整体显著性检验(F检验):")
print(f"   - F统计量: {model.fvalue:.4f}")
print(f"   - F检验p值: {model.f_pvalue:.6f}")

if model.f_pvalue < 0.05:
    print("   - 结论: 拒绝原假设(p < 0.05)，模型整体显著")
    print("   - 意义: 身高与腿长之间存在显著的线性关系")
else:
    print("   - 结论: 接受原假设(p ≥ 0.05)，模型不显著")
    print("   - 意义: 身高与腿长之间不存在显著的线性关系")

# t检验部分
print("\n2. 回归系数显著性检验(t检验):")
print("   - 原假设H₀: β=0 (该变量对因变量无显著影响)")
print("   - 备择假设H₁: β≠0 (该变量对因变量有显著影响)")

for i, coef in enumerate(model.params):
    pvalue = model.pvalues[i]
    t_stat = model.tvalues[i]
    sig = "显著" if pvalue < 0.05 else "不显著"
    
    var_name = "截距(β₀)" if i == 0 else "斜率(β₁)"
    print(f"\n   {var_name}检验:")
    print(f"     - t统计量: {t_stat:.4f}")
    print(f"     - p值: {pvalue:.6f}")
    print(f"     - 结论: {var_name} {sig}")
    
    if i == 0:
        if pvalue < 0.05:
            print("     - 意义: 当身高为0时，腿长显著不为0")
        else:
            print("     - 意义: 当身高为0时，腿长与0无显著差异")
    else:
        if pvalue < 0.05:
            print("     - 意义: 身高对腿长有显著影响")
            print(f"     - 解释: 身高每增加1厘米，腿长平均增加 {coef:.4f} 厘米")
        else:
            print("     - 意义: 身高对腿长无显著影响")

# 模型诊断总结
print("\n3. 模型诊断总结:")
if model.f_pvalue < 0.05 and model.pvalues[1] < 0.05:
    print("   - 模型整体和系数均显著，身高可有效解释腿长变化")
    print(f"   - 身高可解释腿长变异的{r_squared*100:.1f}%")
elif model.f_pvalue >= 0.05:
    print("   - 模型整体不显著，需考虑其他解释变量或非线性关系")
else:
    print("   - 模型整体显著但斜率不显著，可能存在模型设定问题")

# ===== 输出所有预测区间 =====
print("\n===== 所有样本点的预测值和预测区间 =====")
print("身高(cm)\t预测值(cm)\t95%预测区间(cm)")
for xi in x:
    Xi = [1, xi]
    yi_pred = model.predict(Xi)[0]
    pred = model.get_prediction(Xi)
    pred_int = pred.conf_int(alpha=0.05)[0]
    print(f"{xi}\t\t{yi_pred:.3f}\t\t[{pred_int[0]:.3f}, {pred_int[1]:.3f}]")

# ===== 生成预测范围数据（用于后续绘图）=====
x_range = np.linspace(min(x)-5, max(x)+5, 100)
X_range = sm.add_constant(x_range)
predictions = model.get_prediction(X_range)

# 计算置信区间和预测区间
mean_ci = predictions.conf_int(alpha=0.05)  # 95%置信区间
se_obs = np.sqrt(mse_resid + predictions.se_obs**2)
t_value = stats.t.ppf(1 - 0.025, dof_resid)
pred_lower = predictions.predicted - t_value * se_obs
pred_upper = predictions.predicted + t_value * se_obs

# ===== 现在开始绘制所有图片 =====

# 1. 绘制残差图
plt.figure(figsize=(10, 6))
plt.axhline(y=0, color='gray', linestyle='-', alpha=0.7)  # 基准线

# 设置置信区间（±2个标准差）
plt.axhline(y=2, color='red', linestyle='--', alpha=0.7)
plt.axhline(y=-2, color='red', linestyle='--', alpha=0.7)
plt.fill_between(range(len(Y)), -2, 2, color='red', alpha=0.1, label='95%置信区间')

# 绘制残差
plt.scatter(range(1, len(Y)+1), std_residuals, s=100, edgecolor='black', label='标准化残差')
plt.xlabel('样本序号')
plt.ylabel('标准化残差')
plt.title('标准化残差图（异常值检测）', fontsize=14)
plt.legend(loc='best')
plt.grid(True, linestyle='--', alpha=0.5)

# 标注异常点
for i in outliers:
    plt.annotate(f'异常点{i}', (i, std_residuals[i-1]), 
                 xytext=(10, 15 if std_residuals[i-1] > 0 else -25),
                 textcoords='offset points',
                 arrowprops=dict(arrowstyle="->", color='red', alpha=0.7))

plt.tight_layout()
plt.show()

# 2. 绘制回归可视化图
plt.figure(figsize=(12, 8))

# 原始数据散点图
plt.scatter(x, Y, c='dodgerblue', marker='o', s=100, 
            edgecolor='black', alpha=0.8, label='观测值')

# 回归线
plt.plot(x_range, predictions.predicted, 'r-', linewidth=3, 
         label=f'回归线: Y = {b[0]:.1f} + {b[1]:.3f}X')

# 置信区间
plt.fill_between(x_range, mean_ci[:, 0], mean_ci[:, 1], 
                 color='orange', alpha=0.3, label='95%置信区间 (均值)')

# 预测区间
plt.fill_between(x_range, pred_lower, pred_upper, 
                 color='green', alpha=0.2, label='95%预测区间 (个体)')

# 添加回归方程和统计信息
plt.text(0.02, 0.95, f'$Y = {b[0]:.1f} + {b[1]:.3f}X$', 
         transform=plt.gca().transAxes, fontsize=14)
plt.text(0.02, 0.90, f'$R^2 = {r_squared:.4f}$', 
         transform=plt.gca().transAxes, fontsize=14)
plt.text(0.02, 0.85, f'SEE = {see:.3f} cm', 
         transform=plt.gca().transAxes, fontsize=12)
plt.text(0.02, 0.80, f'样本量: n = {len(x)}', 
         transform=plt.gca().transAxes, fontsize=12)

# 添加图例和标签
plt.xlabel('身高 (cm)', fontsize=12)
plt.ylabel('腿长 (cm)', fontsize=12)
plt.title('身高与腿长的一元线性回归分析', fontsize=16)
plt.legend(loc='upper left', frameon=True, shadow=True)
plt.grid(True, linestyle='--', alpha=0.3)
plt.tight_layout()

# 添加区间说明
plt.figtext(0.5, 0.01, 
            "注：置信区间(橙色)表示回归线的不确定性，预测区间(绿色)表示个体预测值的不确定性", 
            ha="center", fontsize=10, style='italic')

plt.show()