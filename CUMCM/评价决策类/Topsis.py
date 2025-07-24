import numpy as np
import pandas as pd

# ======================== 指标类型转换函数 ========================
def min_to_max(max_val, values):
    """
    步骤1.1: 极小型→极大型转换
    原理: 使用最大值差值法 (转换公式: 转换值 = 最大值 - 原始值)
    """
    # 将输入值转换为列表形式
    values = list(values)
    # 对每个值应用转换公式：max_val - val
    converted = [[max_val - val] for val in values]
    # 将结果转换为NumPy数组并返回
    return np.array(converted)

def mid_to_max(optimal_val, values):
    """
    步骤1.2: 中间型→极大型转换
    原理: 基于与最优值的偏差比例 (转换公式: 1 - |值-最优值|/最大偏差)
    """
    # 将输入值转换为列表形式
    values = list(values)
    # 计算每个值与最优值的绝对偏差
    deviations = [abs(val - optimal_val) for val in values]
    # 获取最大偏差值，若为0则设为1避免除零错误
    max_deviation = max(deviations) if max(deviations) != 0 else 1
    # 应用转换公式：1 - (偏差值/最大偏差)
    converted = [[1 - dev / max_deviation] for dev in deviations]
    # 将结果转换为NumPy数组并返回
    return np.array(converted)

def range_to_max(lower_bound, upper_bound, values):
    """
    步骤1.3: 区间型→极大型转换
    原理: 根据值在区间内外计算得分
    - 区间内: 得分为1
    - 低于下限: 1 - (下限-值)/最大偏差
    - 高于上限: 1 - (值-上限)/最大偏差
    """
    # 将输入值转换为列表形式
    values = list(values)
    # 计算最大可能偏差（最小值到下限或上限到最大值）
    max_deviation = max(lower_bound - min(values), max(values) - upper_bound)
    # 若最大偏差为0则设为1避免除零错误
    max_deviation = max_deviation if max_deviation != 0 else 1
    
    converted = []
    # 遍历每个值并计算得分
    for val in values:
        if val < lower_bound:
            # 低于下限的转换公式
            score = 1 - (lower_bound - val) / max_deviation
        elif val > upper_bound:
            # 高于上限的转换公式
            score = 1 - (val - upper_bound) / max_deviation
        else:
            # 区间内得分为1
            score = 1
        converted.append([score])
    # 将结果转换为NumPy数组并返回
    return np.array(converted)

# ======================== 结果输出函数 ========================
def print_formatted_matrix(matrix, title):
    """步骤7.1: 矩阵美化输出 (保留6位小数)"""
    # 打印标题
    print(f"\n{title}")
    # 将NumPy矩阵转换为Pandas DataFrame
    df = pd.DataFrame(matrix)
    # 格式化输出：不显示索引和列名，保留6位小数
    print(df.to_string(index=False, header=False, float_format=lambda x: f"{x:.6f}"))
    
def print_formatted_vector(vector, title, labels=None):
    """步骤7.2: 向量美化输出 (带对象标签)"""
    # 打印标题
    print(f"\n{title}")
    if labels:
        # 如果有标签，创建带标签的DataFrame
        df = pd.DataFrame({"对象": labels, "值": vector})
        # 格式化输出：不显示索引，保留6位小数
        print(df.to_string(index=False, float_format=lambda x: f"{x:.6f}"))
    else:
        # 无标签时创建普通DataFrame
        df = pd.DataFrame({"值": vector})
        # 格式化输出：不显示索引和列名，保留6位小数
        print(df.to_string(index=False, header=False, float_format=lambda x: f"{x:.6f}"))

# ======================== 主函数 (TOPSIS算法流程) ========================
def main():
    # 步骤0: 程序入口
    # 获取评价对象和指标数量
    n = int(input("请输入参评对象数目: "))
    m = int(input("请输入评价指标数目: "))
    
    # 显示指标类型说明
    print("\n指标类型说明:")
    print("1: 极大型（数值越大越好）")
    print("2: 极小型（数值越小越好）")
    print("3: 中间型（数值越接近某个值越好）")
    print("4: 区间型（数值在某个区间内最好）")
    
    # 获取各指标类型输入
    types = input("请输入各指标类型（以空格分隔）: ").split()
    print(f"指标类型向量: {types}")
    
    # 生成默认指标和对象名称
    metrics = [f"指标{i+1}" for i in range(m)]
    objects = [f"对象{i+1}" for i in range(n)]
    
    # 步骤1: 获取原始评价矩阵
    print("\n请输入评价矩阵（每行一个对象，各指标值用空格分隔）:")
    matrix = np.zeros((n, m))  # 创建n×m的零矩阵
    for i in range(n):
        # 获取每个对象的指标值
        row = list(map(float, input(f"对象{i+1}: ").split()))
        matrix[i] = row  # 将输入值赋给矩阵的对应行
    
    # 输出格式化后的原始评价矩阵
    print_formatted_matrix(matrix, "【步骤1】原始评价矩阵")
    
    # 步骤2: 指标类型统一转换 (全部转为极大型指标)
    unified_matrix = np.zeros((n, 1))  # 初始化转换后的矩阵
    
    # 遍历每个指标进行类型转换
    for col in range(m):
        metric_values = matrix[:, col]  # 提取当前列的所有值
        
        # 根据指标类型选择转换方法
        if types[col] == "1":
            # 极大型指标无需转换
            converted = np.array([[v] for v in metric_values])
        elif types[col] == "2":
            # 极小型转换
            max_val = max(metric_values)
            converted = min_to_max(max_val, metric_values)
            print(f"\n已将指标{col+1}（极小型）转换为极大型指标")
        elif types[col] == "3":
            # 中间型转换
            optimal = float(input(f"\n请输入指标{col+1}（中间型）的最优值: "))
            converted = mid_to_max(optimal, metric_values)
            print(f"已将指标{col+1}（中间型）转换为极大型指标，最优值为: {optimal}")
        elif types[col] == "4":
            # 区间型转换
            print(f"\n请输入指标{col+1}（区间型）的最优区间:")
            lower = float(input("区间下限: "))
            upper = float(input("区间上限: "))
            converted = range_to_max(lower, upper, metric_values)
            print(f"已将指标{col+1}（区间型）转换为极大型指标，最优区间: [{lower}, {upper}]")
        
        # 将转换后的列合并到统一矩阵
        if col == 0:
            unified_matrix = converted
        else:
            unified_matrix = np.hstack([unified_matrix, converted])
    
    # 输出统一类型后的矩阵
    print_formatted_matrix(unified_matrix, "\n【步骤2】统一指标类型后的矩阵")
    
    # 步骤3: 标准化处理 (向量归一化)
    """
    标准化公式：
        z_{ij} = x_{ij} / √(Σx_{kj}²) 
    目的: 消除量纲影响，使不同指标具有可比性
    """
    norms = np.sqrt(np.sum(unified_matrix**2, axis=0))  # 计算每列的范数（平方和的平方根）
    standardized = unified_matrix / norms  # 应用归一化公式
    print_formatted_matrix(standardized, "\n【步骤3】标准化矩阵")
    
    # 步骤4: 确定理想解
    """
    正理想解(V+): 各指标的最大值 (最优情况)
    负理想解(V-): 各指标的最小值 (最劣情况)
    """
    ideal_best = standardized.max(axis=0)  # 每列最大值（正理想解）
    ideal_worst = standardized.min(axis=0)  # 每列最小值（负理想解）
    
    # 输出理想解
    print_formatted_vector(ideal_best, "\n【步骤4】理想最优解（各指标最大值）", metrics)
    print_formatted_vector(ideal_worst, "\n【步骤4】理想最劣解（各指标最小值）", metrics)
    
    # 步骤5: 距离计算 (欧氏距离)
    """
    距离公式:
        D+ = √[Σ(v_ij - V+_j)²] (与最优解的距离)
        D- = √[Σ(v_ij - V-_j)²] (与最劣解的距离)
    """
    # 计算每个对象与正理想解的欧氏距离
    dist_best = np.sqrt(np.sum((standardized - ideal_best)**2, axis=1))
    # 计算每个对象与负理想解的欧氏距离
    dist_worst = np.sqrt(np.sum((standardized - ideal_worst)**2, axis=1))
    
    # 输出距离计算结果
    print_formatted_vector(dist_best, "\n【步骤5】对象与理想最优解的距离(d+)", objects)
    print_formatted_vector(dist_worst, "\n【步骤5】对象与理想最劣解的距离(d-)", objects)
    
    # 步骤6: 计算得分并排序
    """
    相对接近度公式:
        C_i = D- / (D+ + D-)
    意义: 值越大表示方案越接近理想解
    """
    # 计算每个对象的相对接近度得分（0-100范围）
    scores = 100 * dist_worst / (dist_best + dist_worst)
    # 按得分降序排列的索引
    sorted_indices = np.argsort(scores)[::-1]
    
    # 步骤7: 结果输出
    print("\n【步骤6】综合评价结果")
    results = []
    for rank, idx in enumerate(sorted_indices, 1):
        # 计算标准化得分（占总分的百分比）
        normalized_score = f"{100*scores[idx]/sum(scores):.2f}%"
        # 收集结果数据
        results.append({
            "排名": rank,
            "对象": objects[idx],
            "得分": scores[idx],
            "标准化得分": normalized_score
        })
    
    # 创建结果DataFrame并按格式输出
    results_df = pd.DataFrame(results)
    print(results_df.to_string(index=False, formatters={
        "得分": lambda x: f"{x:.6f}",         # 得分保留6位小数
        "标准化得分": lambda x: x             # 标准化得分保持原样
    }))

# 程序执行入口
if __name__ == "__main__":
    main()