import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import itertools  # 用于特征组合
import mlxtend  # 确保已安装mlxtend==0.21.0
print(f"mlxtend版本: {mlxtend.__version__}")  # 确认版本为0.21.0

# ===== 解决中文显示问题 =====
plt.rcParams['font.sans-serif'] = ['SimHei']  
plt.rcParams['axes.unicode_minus'] = False

# ==================== 步骤1: 数据准备 ====================
# 根据文档表7-1创建数据集 (建材销售量)
data = {
    '推销开支': [5.5, 2.5, 8.0, 3.0, 3.0, 2.9, 8.0, 9.0, 4.0, 6.5, 5.5, 5.0, 6.0, 5.0, 3.5, 8.0, 6.0, 4.0, 7.5, 7.0],
    '实际账目数': [31, 55, 67, 50, 38, 71, 30, 56, 42, 73, 60, 44, 50, 39, 55, 70, 40, 50, 62, 59],
    '同类竞争数': [10, 8, 12, 7, 8, 12, 12, 5, 8, 5, 11, 12, 6, 10, 10, 6, 11, 11, 9, 9],
    '销售潜力': [8, 6, 9, 16, 15, 17, 8, 10, 4, 16, 7, 12, 6, 4, 4, 14, 6, 8, 13, 11],
    '建材销量': [79.3, 200.1, 163.2, 200.1, 146.0, 177.7, 30.9, 291.9, 160.0, 339.4, 159.6, 86.3, 237.5, 107.2, 155.0, 201.4, 100.2, 135.8, 223.3, 195.0]
}

df = pd.DataFrame(data)
X = df[['推销开支', '实际账目数', '同类竞争数', '销售潜力']]
y = df['建材销量']

# ==================== 步骤2: 初始多元线性回归 ====================
# 添加常数项 (截距)
X_const = sm.add_constant(X)

# 创建并训练模型
model_full = sm.OLS(y, X_const).fit()

# 打印完整回归结果
print("="*50)
print("初始多元线性回归结果 (所有变量)")
print("="*50)
print(model_full.summary())

# ==================== 步骤3: 逐步回归优化 ====================
print("\n" + "="*50)
print("逐步回归优化过程")
print("="*50)

# 手动实现逐步回归过程
def manual_stepwise_selection(X, y):
    """手动实现逐步回归过程"""
    included = []
    excluded = list(X.columns)
    history = []
    
    # 初始模型 - 仅截距
    current_score, best_new_score = float('inf'), float('inf')
    while excluded and current_score == best_new_score:
        scores_with_candidates = []
        # 尝试添加每个特征
        for candidate in excluded:
            features = included + [candidate]
            X_temp = sm.add_constant(X[features])
            model = sm.OLS(y, X_temp).fit()
            score = model.ssr  # 残差平方和 (越小越好)
            scores_with_candidates.append((score, candidate, features))
        
        # 按残差平方和排序
        scores_with_candidates.sort(key=lambda x: x[0])
        best_new_score, best_candidate, best_features = scores_with_candidates[0]
        
        # 检查是否改善模型
        if not included or best_new_score < current_score:
            included.append(best_candidate)
            excluded.remove(best_candidate)
            current_score = best_new_score
            r2 = sm.OLS(y, sm.add_constant(X[included])).fit().rsquared
            history.append((len(included), best_candidate, r2, best_features.copy()))
            print(f"步骤 {len(included)}: 添加特征 '{best_candidate}' | R² = {r2:.4f}")
        else:
            break
    
    # 尝试移除特征
    print("\n尝试移除不显著特征...")
    improved = True
    while improved and len(included) > 1:
        improved = False
        for candidate in included:
            if candidate == 'const': continue
            features = [f for f in included if f != candidate]
            X_temp = sm.add_constant(X[features])
            model = sm.OLS(y, X_temp).fit()
            if model.rsquared >= current_score:  # 移除不会降低R²
                included.remove(candidate)
                current_score = model.rsquared
                improved = True
                r2 = current_score
                history.append((len(included), f"移除 {candidate}", r2, features.copy()))
                print(f"步骤 {len(history)}: 移除特征 '{candidate}' | R² = {r2:.4f}")
                break
    
    return included, history

# 执行逐步回归
selected_features, history = manual_stepwise_selection(X, y)
print("\n最终选择特征:", selected_features)

# 可视化逐步回归过程
plt.figure(figsize=(12, 6))
steps = [f"步骤 {i+1}\n{desc}" for i, (step, desc, r2, feats) in enumerate(history)]
r2_values = [r2 for _, _, r2, _ in history]
features_count = [len(feats) for _, _, _, feats in history]

plt.subplot(1, 2, 1)
plt.plot(steps, r2_values, 'bo-')
plt.xlabel('回归步骤')
plt.ylabel('R²值')
plt.title('逐步回归过程中R²的变化')
plt.xticks(rotation=45, ha='right')
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
plt.plot(steps, features_count, 'go-')
plt.xlabel('回归步骤')
plt.ylabel('特征数量')
plt.title('逐步回归过程中特征数量的变化')
plt.xticks(rotation=45, ha='right')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('stepwise_process.png')
plt.show()

# ==================== 步骤4: 最优模型建立 ====================
# 使用最优特征集
X_optimal = sm.add_constant(X[selected_features])
model_optimal = sm.OLS(y, X_optimal).fit()

# 打印最优模型结果
print("\n" + "="*50)
print("最优回归模型结果")
print("="*50)
print(model_optimal.summary())

# ==================== 步骤5: 残差分析 ====================
plt.figure(figsize=(15, 10))

# 1. 残差与拟合值关系图
plt.subplot(2, 2, 1)
plt.scatter(model_optimal.fittedvalues, model_optimal.resid, alpha=0.7)
plt.axhline(y=0, color='r', linestyle='--')
plt.xlabel('拟合值')
plt.ylabel('残差')
plt.title('残差 vs 拟合值')
plt.grid(True, alpha=0.3)

# 2. 残差QQ图
plt.subplot(2, 2, 2)
sm.qqplot(model_optimal.resid, line='s', ax=plt.gca())
plt.title('残差正态概率图(QQ图)')

# 3. 残差直方图
plt.subplot(2, 2, 3)
plt.hist(model_optimal.resid, bins=15, edgecolor='k', alpha=0.7)
plt.xlabel('残差')
plt.ylabel('频数')
plt.title('残差分布直方图')
plt.grid(True, alpha=0.3)

# 4. 特征重要性
plt.subplot(2, 2, 4)
importance = model_optimal.params.drop('const').abs().sort_values(ascending=False)
importance.plot(kind='bar')
plt.title('特征重要性(系数绝对值)')
plt.ylabel('系数值')
plt.xticks(rotation=45)

plt.tight_layout()
plt.savefig('residual_analysis.png')
plt.show()

# ==================== 步骤6: 置信区间和预测区间 ====================
# 获取预测值
y_pred = model_optimal.predict(X_optimal)

# 计算置信区间和预测区间
prediction = model_optimal.get_prediction(X_optimal)
ci = prediction.conf_int(alpha=0.05)  # 95%置信区间
pi_lower = y_pred - 1.96 * np.sqrt(model_optimal.scale)  # 预测区间下限
pi_upper = y_pred + 1.96 * np.sqrt(model_optimal.scale)  # 预测区间上限

# 可视化区间估计
plt.figure(figsize=(12, 8))
index = np.arange(len(y))

plt.plot(index, y, 'bo', label='实际值')
plt.plot(index, y_pred, 'r-', label='预测值')
plt.fill_between(index, ci[:, 0], ci[:, 1], color='gray', alpha=0.3, label='95%置信区间')
plt.fill_between(index, pi_lower, pi_upper, color='yellow', alpha=0.2, label='95%预测区间')

plt.xlabel('样本序号')
plt.ylabel('建材销售量')
plt.title('实际值、预测值及区间估计')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('prediction_intervals.png')
plt.show()

# ==================== 步骤7: 拟合优度评估 ====================
# 计算R²和调整R²
r2 = model_optimal.rsquared
adj_r2 = model_optimal.rsquared_adj

# 计算均方根误差(RMSE)
rmse = np.sqrt(model_optimal.mse_resid)

# 拟合优度可视化
plt.figure(figsize=(10, 6))
plt.scatter(y, y_pred, edgecolor='k', alpha=0.7)
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', lw=2)
plt.xlabel('实际值')
plt.ylabel('预测值')
plt.title(f'拟合优度评估 (R² = {r2:.4f}, RMSE = {rmse:.2f})')
plt.grid(True, alpha=0.3)

# 添加文本标注
textstr = '\n'.join((
    f'R² = {r2:.4f}',
    f'调整R² = {adj_r2:.4f}',
    f'RMSE = {rmse:.2f}',
    f'样本数 = {len(y)}'))
plt.gcf().text(0.15, 0.85, textstr, fontsize=12, bbox=dict(facecolor='white', alpha=0.5))

plt.tight_layout()
plt.savefig('goodness_of_fit.png')
plt.show()

# ==================== 步骤8: 点预测和区间预测 ====================
# 创建新样本进行预测
new_data = pd.DataFrame({
    '推销开支': [6.0, 4.5, 7.8],
    '实际账目数': [45, 60, 50],
    '同类竞争数': [8, 10, 9],
    '销售潜力': [10, 12, 15]
})

# 仅使用最优特征
if 'const' in X_optimal.columns:
    X_new = sm.add_constant(new_data[selected_features])
else:
    X_new = new_data[selected_features]

# 点预测
predictions = model_optimal.predict(X_new)

# 区间预测
prediction_results = model_optimal.get_prediction(X_new)
pred_int = prediction_results.conf_int(alpha=0.05)  # 95%预测区间

# 打印预测结果
print("\n" + "="*50)
print("点预测和区间预测结果")
print("="*50)
for i, (point, lower, upper) in enumerate(zip(predictions, pred_int[:, 0], pred_int[:, 1])):
    print(f"样本 {i+1}:")
    print(f"  点预测值: {point:.2f}")
    print(f"  95%预测区间: [{lower:.2f}, {upper:.2f}]")
    print(f"  区间宽度: {upper-lower:.2f}\n")