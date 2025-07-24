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