import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ===== 解决中文显示问题 =====
plt.rcParams['font.sans-serif'] = ['SimHei']  
plt.rcParams['axes.unicode_minus'] = False

def grey_predict(data, predict_num=1):
    """
    灰色预测GM(1,1)模型
    :param data: 原始非负时间序列数据（1维数组或列表）
    :param predict_num: 预测未来多少期
    :return: 包含拟合值和预测值的数组，以及模型参数和检验指标
    """
    # ================= 1. 数据预处理 =================
    x0 = np.array(data, dtype=np.float64)
    n = len(x0)
    years = list(range(1, n+1))  # 创建时间序列
    
    # 检查数据长度
    if n < 4:
        raise ValueError("数据长度至少需要4个点")
    
    # ================= 2. 绘制原始数据时间序列图 =================
    plt.figure(figsize=(10, 5))
    plt.plot(years, x0, 'o-', linewidth=2, markersize=8)
    plt.title('原始数据时间序列图')
    plt.xlabel('时间序列')
    plt.ylabel('数值')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.xticks(years)
    plt.tight_layout()
    plt.show()
    
    # ================= 3. 级比检验（光滑比检验） =================
    # 计算光滑比 ρ(k) = x0(k)/[x0(1)+x0(2)+...+x0(k-1)]
    rho = np.zeros(n)
    rho[0] = np.nan  # 第一个数据无光滑比
    for k in range(1, n):
        rho[k] = x0[k] / np.sum(x0[:k])
    
    # 计算合格率（除前两个外，其余光滑比<0.5的比例）
    valid_ratio = np.sum(rho[2:] < 0.5) / (n - 2)
    
    print("="*50)
    print("级比检验结果:")
    print(f"光滑比序列: {np.round(rho, 4)}")
    print(f"光滑比合格率: {valid_ratio:.2%} (要求≥90%)")
    
    if valid_ratio < 0.9:
        print("警告：光滑比检验未通过！可能不适合灰色预测模型")
    
    # ================= 4. 绘制光滑度图形 =================
    plt.figure(figsize=(10, 5))
    # 绘制光滑比曲线
    plt.plot(years[1:], rho[1:], 's-', color='dodgerblue', linewidth=2, markersize=8, label='光滑比')
    # 绘制参考线
    plt.axhline(y=0.5, color='r', linestyle='--', alpha=0.7, label='临界值(0.5)')
    # 填充合格区域
    plt.fill_between(years[1:], 0, 0.5, color='green', alpha=0.1)
    
    # 标注合格率
    plt.text(years[-1], 0.1, f'合格率: {valid_ratio:.2%}', 
             fontsize=12, ha='right', va='center', 
             bbox=dict(facecolor='white', alpha=0.8))
    
    plt.title('数据光滑度检验')
    plt.xlabel('时间序列')
    plt.ylabel('光滑比(ρ)')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.xticks(years[1:])
    plt.ylim(0, max(rho[1:]) * 1.1)  # 设置Y轴范围
    plt.tight_layout()
    plt.show()
    
    # ================= 5. 累加生成序列（1-AGO） =================
    x1 = np.cumsum(x0)
    
    # ================= 6. 构造数据矩阵B和常数向量Y =================
    # 计算紧邻均值生成序列z1
    z1 = (x1[:-1] + x1[1:]) / 2.0
    
    B = np.vstack([-z1, np.ones(len(z1))]).T
    Y = x0[1:].reshape(-1, 1)
    
    # ================= 7. 最小二乘法求解参数 =================
    # 求解方程：a为发展系数，u为灰色作用量
    a, u = np.linalg.lstsq(B, Y, rcond=None)[0]
    a, u = a[0], u[0]  # 提取标量值
    
    print("="*50)
    print("模型参数估计结果:")
    print(f"发展系数 a = {a:.6f}")
    print(f"灰色作用量 u = {u:.6f}")
    
    # ================= 8. 建立预测模型 =================
    # 累加序列预测公式：x1(k+1) = (x0(1)-u/a)e^{-ak} + u/a
    def x1_predict(k):
        return (x0[0] - u/a) * np.exp(-a * k) + u/a
    
    # 原始序列预测公式：x0(k) = x1(k) - x1(k-1)
    def x0_predict(k):
        if k == 0:
            return x0[0]
        return x1_predict(k) - x1_predict(k-1)
    
    # ================= 9. 计算拟合值 =================
    # 历史拟合值（包括第一个点）
    fitted_vals = np.array([x0_predict(k) for k in range(n)])
    
    # ================= 10. 模型检验 =================
    # 残差检验
    residuals = x0 - fitted_vals
    relative_errors = np.abs(residuals) / x0
    avg_relative_error = np.mean(relative_errors[1:])  # 从第二个点开始计算
    
    # 级比偏差检验
    ratios = x0[1:] / x0[:-1]  # 原始级比σ(k)
    # 理论级比σ'(k) = (1-0.5a)/(1+0.5a)
    theory_ratio = (1 - 0.5*a) / (1 + 0.5*a)
    ratio_deviations = np.abs(1 - theory_ratio / ratios)
    avg_ratio_deviation = np.mean(ratio_deviations)
    
    print("="*50)
    print("模型检验结果:")
    print(f"平均相对误差: {avg_relative_error:.2%} (要求<20%)")
    print(f"平均级比偏差: {avg_ratio_deviation:.4f} (要求<0.2)")
    
    # ================= 11. 预测未来值 =================
    future_points = []
    for i in range(1, predict_num+1):
        future_points.append(x0_predict(n-1+i))
    
    # 合并拟合和预测结果
    all_predicted = np.concatenate([fitted_vals, future_points])
    
    # ================= 12. 结果可视化 =================
    plt.figure(figsize=(10, 6))
    # 绘制原始数据
    plt.plot(years, x0, 'o-', color='blue', linewidth=2, markersize=8, label='原始数据')
    # 绘制拟合值
    plt.plot(years, fitted_vals, 's--', color='green', linewidth=2, markersize=8, label='拟合值')
    
    # 如果有预测值，绘制预测值
    if predict_num > 0:
        predict_years = list(range(n+1, n+predict_num+1))
        plt.plot(predict_years, future_points, '*--', color='red', linewidth=2, markersize=10, label='预测值')
    
    # 添加标题和标签
    plt.title('灰色预测GM(1,1)模型结果')
    plt.xlabel('时间序列')
    plt.ylabel('数值')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # 添加残差信息
    plt.text(years[-1], min(x0)*0.8, 
             f'平均相对误差: {avg_relative_error:.2%}\n平均级比偏差: {avg_ratio_deviation:.4f}',
             fontsize=10, ha='right', bbox=dict(facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.show()
    
    return {
        'fitted_values': fitted_vals,
        'predict_values': np.array(future_points),
        'parameters': {'a': a, 'u': u},
        'tests': {
            'rho': rho,
            'valid_ratio': valid_ratio,
            'avg_relative_error': avg_relative_error,
            'avg_ratio_deviation': avg_ratio_deviation
        }
    }

# ================= 示例使用 =================
if __name__ == "__main__":
    # 长江水质污染数据（1995-2004年废水排放总量）
    data = [174, 179, 183, 189, 207, 234, 220.5, 256, 270, 285]
    
    print("="*50)
    print("灰色预测GM(1,1)模型应用")
    print("="*50)
    print("原始数据:", data)
    
    # 进行灰色预测，预测未来2年
    results = grey_predict(data, predict_num=2)
    
    # 打印预测结果
    print("="*50)
    print("最终预测结果:")
    print(f"历史拟合值: {np.round(results['fitted_values'], 2)}")
    print(f"未来预测值: {np.round(results['predict_values'], 2)}")