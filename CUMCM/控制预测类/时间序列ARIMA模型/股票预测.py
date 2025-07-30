# -*- coding: utf-8 -*-
"""
ARIMA 模型时间序列预测完整流程
--------------------------------
1. 数据准备与可视化
2. 平稳性检验与差分处理
3. 模型参数确定（ACF/PACF + BIC 网格搜索）
4. 模型训练与残差诊断
5. 模型预测与可视化
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller as ADF
import itertools

# ===== 解决中文显示问题 =====
plt.rcParams['font.sans-serif'] = ['SimHei']  
plt.rcParams['axes.unicode_minus'] = False

# ------------------------------------------------------------
# 1. 数据准备与可视化
# ------------------------------------------------------------
# 读入数据，日期列设为索引
ChinaBank = pd.read_csv(r'D:\2025_Summer_Learn\CUMCM\控制预测类\时间序列ARIMA模型\ChinaBank.csv',
                        index_col='Date',
                        parse_dates=['Date'])

# 2014-01 到 2014-06 的收盘价
sub = ChinaBank.loc['2014-01':'2014-06', 'Close']

# 训练集(1-3月) 与 测试集(4-6月)
train = sub.loc['2014-01':'2014-03']
test  = sub.loc['2014-04':'2014-06']

# 画训练集
plt.figure(figsize=(12, 4))
plt.plot(train, label='训练集')
plt.title('招商银行 2014-01 ~ 2014-03 收盘价')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# ------------------------------------------------------------
# 2. 平稳性检验与差分处理
# ------------------------------------------------------------
# 仅取 2014 年 1–3 月数据用于 ADF 和差分
train_close = train

# 差分
train_diff1 = train_close.diff(1)
train_diff2 = train_diff1.diff(1)

# ADF 检验
print('\n========== 基于 1–3 月数据的 ADF 单位根检验 ==========')
for name, series in [
    ('1–3月原始序列', train_close),
    ('1–3月一阶差分', train_diff1.dropna()),
    ('1–3月二阶差分', train_diff2.dropna())
]:
    t_stat, p_val, *_ = ADF(series)
    print(f'{name}: ADF={t_stat:.4f}, p-value={p_val:.4f}')
# ------------------------------------------------------------
# 3. 模型参数确定
# ------------------------------------------------------------
# 3.1 画 ACF/PACF
fig = plt.figure(figsize=(12, 6))
ax1 = fig.add_subplot(311); ax1.plot(train_close); ax1.set_title('1–3月原始序列')
ax2 = fig.add_subplot(312); ax2.plot(train_diff1); ax2.set_title('1–3月一阶差分')
ax3 = fig.add_subplot(313); ax3.plot(train_diff2); ax3.set_title('1–3月二阶差分')
plt.tight_layout()
plt.show()

# 3.2 网格搜索 BIC
p_max = 5; d_max = 0; q_max = 5
results_bic = pd.DataFrame(index=[f'AR{i}' for i in range(p_max+1)],
                           columns=[f'MA{i}' for i in range(q_max+1)])

for p, d, q in itertools.product(range(p_max+1), range(d_max+1), range(q_max+1)):
    if p == d == q == 0:
        results_bic.loc[f'AR{p}', f'MA{q}'] = np.nan
        continue
    try:
        model = sm.tsa.ARIMA(train, order=(p, d, q))
        res = model.fit()
        results_bic.loc[f'AR{p}', f'MA{q}'] = res.bic
    except:
        results_bic.loc[f'AR{p}', f'MA{q}'] = np.nan

results_bic = results_bic.astype(float)
print('\n========== BIC 矩阵 ==========')
print(results_bic)

# 可视化热力图
plt.figure(figsize=(8, 6))
sns.heatmap(results_bic, annot=True, fmt='.2f', cmap='Purples')
plt.title('BIC 值（越小越好）')
plt.tight_layout()
plt.show()

# 最优参数
best_p, best_q = results_bic.stack().idxmin()
best_p = int(best_p.replace('AR', ''))
best_q = int(best_q.replace('MA', ''))
best_d = 0
print(f'\n最优参数 (p,d,q) = ({best_p},{best_d},{best_q})')


# 利用已有模型取最优参数（再次检验p和q的值）
train_results = sm.tsa.arma_order_select_ic(train, ic=['bic'], trend='n', max_ar=8, max_ma=8)

print('BIC', train_results.bic_min_order)
    
# ------------------------------------------------------------
# 4. 模型训练与残差诊断
# ------------------------------------------------------------
model = sm.tsa.ARIMA(train, order=(best_p, best_d, best_q))
results = model.fit()

# 残差 ACF
resid = results.resid
fig, ax = plt.subplots(figsize=(10, 3))
sm.graphics.tsa.plot_acf(resid, lags=30, ax=ax)
plt.title('残差 ACF')
plt.tight_layout()
plt.show()

# ------------------------------------------------------------
# 5. 模型预测与可视化
# ------------------------------------------------------------
# 5.1 训练集内拟合
train_pred = results.predict(dynamic=False)

# 5.2 测试集外推预测
forecast_steps = len(test)
forecast_res = results.get_forecast(steps=forecast_steps)
forecast_mean = forecast_res.predicted_mean
forecast_ci   = forecast_res.conf_int()

# 绘图
plt.figure(figsize=(14, 6))
# 实际
plt.plot(sub, label='实际值')
# 训练集拟合
plt.plot(train.index, train_pred, label='训练集拟合')
# 测试集预测
plt.plot(test.index, forecast_mean, label='测试集预测', linestyle='--')
plt.fill_between(test.index,
                 forecast_ci.iloc[:, 0],
                 forecast_ci.iloc[:, 1],
                 color='pink', alpha=0.3)
# 竖线分隔训练/测试
plt.axvline(test.index[0], color='red', linestyle='--', alpha=0.7)
plt.title('招商银行 2014-01 ~ 2014-06 收盘价 ARIMA 预测')
plt.xticks(rotation=45)
plt.legend()
plt.tight_layout()
plt.show()

print('BIC', train_results.bic_min_order)