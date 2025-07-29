import numpy as np

def print_formatted_matrix(matrix, title=""):
    """打印格式化矩阵，保留4位小数"""
    print(f"\n{title}:")
    for row in matrix:
        print("\t".join([f"{x:.4f}" for x in row]))

# ======================== 指标类型转换函数 ========================
def min_to_max(values):
    """将极小型指标转换为极大型指标"""
    max_val = np.max(values)
    return max_val - values

def mid_to_max(optimal, values):
    """将中间型指标转换为极大型指标"""
    abs_diffs = np.abs(values - optimal)
    max_diff = np.max(abs_diffs)
    # 避免除零错误
    if max_diff == 0:
        return np.ones_like(values)
    return 1 - abs_diffs / max_diff

def range_to_max(lower, upper, values):
    """将区间型指标转换为极大型指标"""
    n = len(values)
    result = np.zeros(n)
    for i in range(n):
        if values[i] < lower:
            result[i] = 1 - (lower - values[i]) / (lower - np.min(values))
        elif values[i] > upper:
            result[i] = 1 - (values[i] - upper) / (np.max(values) - upper)
        else:
            result[i] = 1
    return result

# ======================== 熵权法核心函数 ========================
def handle_zero_log(p):
    """处理对数计算中的零值，避免出现无穷小"""
    result = np.zeros_like(p)  # 创建与输入数组相同形状的零数组
    mask = p > 0  # 创建掩码，标记所有大于零的元素
    result[mask] = np.log(p[mask])  # 仅对大于零的元素计算对数
    return result

def normalize_matrix(X):
    """对矩阵进行列标准化处理（极差标准化）"""
    n, m = X.shape
    Z = np.zeros_like(X, dtype=float)
    
    for j in range(m):
        col = X[:, j]
        min_val = np.min(col)
        max_val = np.max(col)
        range_val = max_val - min_val
        
        # 避免除零错误
        if range_val == 0:
            Z[:, j] = 0.5  # 全相同列设为0.5
        else:
            # 直接进行0-1标准化，不添加额外的线性变换
            Z[:, j] = (col - min_val) / range_val
    
    return Z

def calculate_entropy_weights(Z):
    """计算熵权法权重"""
    n, m = Z.shape  # 获取矩阵的行数(样本数)和列数(指标数)
    
    # 初始化结果数组
    entropy = np.zeros(m)       # 存储各指标的信息熵
    utility = np.zeros(m)       # 存储各指标的信息效用值
    weights = np.zeros(m)       # 存储最终计算的权重
    
    # 步骤1: 计算每个指标的概率分布
    for j in range(m):
        column = Z[:, j]
        # 避免全零列导致的除零错误
        if np.sum(column) == 0:
            p = np.zeros(n)  # 若列为全零，概率设为0
        else:
            p = column / np.sum(column)
        
        # 步骤2: 计算信息熵
        # 使用自定义函数处理对数计算中的零值
        log_p = handle_zero_log(p)
        entropy[j] = -np.sum(p * log_p) / np.log(n)
        
        # 步骤3: 计算信息效用值 (信息熵的互补值)
        utility[j] = 1 - entropy[j]
    
    # 步骤4: 归一化效用值得到最终权重
    # 避免所有效用值均为零的情况
    total_utility = np.sum(utility)
    if total_utility == 0:
        weights = np.ones(m) / m  # 若所有效用值为零，均匀分配权重
    else:
        weights = utility / total_utility
    
    return weights, entropy, utility

# 主程序
if __name__ == "__main__":
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
    if len(types) != m:
        print(f"错误: 需要{m}个指标类型，但输入了{len(types)}个")
        exit(1)
    
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
        if len(row) != m:
            print(f"错误: 需要{m}个指标值，但输入了{len(row)}个")
            exit(1)
        matrix[i] = row  # 将输入值赋给矩阵的对应行
    
    # 输出格式化后的原始评价矩阵
    print_formatted_matrix(matrix, "【步骤1】原始评价矩阵")
    
    # 步骤2: 指标类型统一转换 (全部转为极大型指标)
    unified_matrix = np.zeros((n, m))  # 初始化转换后的矩阵
    
    # 遍历每个指标进行类型转换
    for col in range(m):
        metric_values = matrix[:, col]  # 提取当前列的所有值
        
        # 根据指标类型选择转换方法
        if types[col] == "1":
            # 极大型指标无需转换
            unified_matrix[:, col] = metric_values
        elif types[col] == "2":
            # 极小型转换
            unified_matrix[:, col] = min_to_max(metric_values)
            print(f"已将指标{col+1}（极小型）转换为极大型指标")
        elif types[col] == "3":
            # 中间型转换
            optimal = float(input(f"\n请输入指标{col+1}（中间型）的最优值: "))
            unified_matrix[:, col] = mid_to_max(optimal, metric_values)
            print(f"已将指标{col+1}（中间型）转换为极大型指标，最优值为: {optimal}")
        elif types[col] == "4":
            # 区间型转换
            print(f"\n请输入指标{col+1}（区间型）的最优区间:")
            lower = float(input("区间下限: "))
            upper = float(input("区间上限: "))
            unified_matrix[:, col] = range_to_max(lower, upper, metric_values)
            print(f"已将指标{col+1}（区间型）转换为极大型指标，最优区间: [{lower}, {upper}]")
        else:
            print(f"错误: 未知指标类型 '{types[col]}'")
            exit(1)
    
    # 输出统一类型后的矩阵
    print_formatted_matrix(unified_matrix, "\n【步骤2】统一指标类型后的矩阵")
    
    # 步骤3: 数据标准化（使用极差标准化）
    Z = normalize_matrix(unified_matrix)
    print_formatted_matrix(Z, "\n【步骤3】标准化矩阵")
    
    # 步骤4: 计算熵权
    W, entropy, utility = calculate_entropy_weights(Z)
    
    # 步骤5: 输出结果
    print("\n【计算结果】")
    print("信息熵 =", np.round(entropy, 4))
    print("信息效用值 =", np.round(utility, 4))
    print("最终权重 =", np.round(W, 4))
    print("权重之和 =", round(np.sum(W), 6))  # 验证权重之和是否为1
    
    # 计算综合得分
    scores = Z @ W
    ranked_indices = np.argsort(-scores)  # 从高到低排序的索引
    
    print("\n【综合评价结果】")
    print("对象\t综合得分\t排名")
    for i, idx in enumerate(ranked_indices):
        print(f"{objects[idx]}\t{scores[idx]:.4f}\t{i+1}")