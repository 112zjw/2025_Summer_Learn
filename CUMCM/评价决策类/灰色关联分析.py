import numpy as np

def input_matrix():
    """获取用户输入的评估矩阵和指标类型"""
    print("|| 灰色关联分析系统 ||")
    print("> 步骤1/4：输入参评对象数量（行数）")
    n = int(input(">>> 参评数目: "))
    print("> 步骤2/4：输入评价指标数量（列数）")
    m = int(input(">>> 指标数目: "))
    
    print("> 步骤3/4：设定指标类型（共4类）")
    print("  [1]极大型(越大越好) [2]极小型(越小越好)")
    print("  [3]中间型(接近某值) [4]区间型(在范围内)")
    print(f"请输入{m}个类型编号（空格分隔）：")
    kind = input(">>> 类型矩阵: ").split()
    
    print("> 步骤4/4：输入评估数据矩阵")
    print(f"格式：输入{n}行数据，每行{m}个数值（空格分隔）")
    A = np.zeros(shape=(n, m))
    for i in range(n):
        row = list(map(float, input(f"• 第{i+1}行数据: ").split()))
        A[i] = row
    
    print("\n= 输入矩阵 =")
    print(A)
    return n, m, kind, A

def minTomax(maxx, x):
    """极小型指标→极大型指标"""
    x = list(x)
    ans = [[(maxx - e)] for e in x]
    return np.array(ans)

def midTomax(bestx, x):
    """中间型指标→极大型指标"""
    x = list(x)
    h = [abs(e - bestx) for e in x]
    M = max(h)
    if M == 0:
        M = 1
    ans = [[(1 - e / M)] for e in h]
    return np.array(ans)

def regTomax(lowx, highx, x):
    """区间型指标→极大型指标"""
    x = list(x)
    M = max(lowx - min(x), max(x) - highx)
    if M == 0:
        M = 1
    ans = []
    for i in range(len(x)):
        if x[i] < lowx:
            ans.append([(1 - (lowx - x[i]) / M)])
        elif x[i] > highx:
            ans.append([(1 - (x[i] - highx) / M)])
        else:
            ans.append([1])
    return np.array(ans)

def unify_indicators(n, m, kind, A):
    """统一指标类型为极大型"""
    X = np.zeros(shape=(n, 1))
    for i in range(m):
        if kind[i] == "1":
            v = np.array(A[:, i])
        elif kind[i] == "2":
            maxA = max(A[:, i])
            v = minTomax(maxA, A[:, i])
        elif kind[i] == "3":
            print("类型三：请输入最优值：")
            bestA = eval(input())
            v = midTomax(bestA, A[:, i])
        elif kind[i] == "4":
            print("类型四：请输入区间[a, b]值a：")
            lowA = eval(input())
            print("类型四：请输入区间[a, b]值b：")
            highA = eval(input())
            v = regTomax(lowA, highA, A[:, i])
        if i == 0:
            X = v.reshape(-1, 1)
        else:
            X = np.hstack([X, v.reshape(-1, 1)])
    print("统一指标后矩阵为：\n{}".format(X))
    return X

def preprocess_matrix(X):
    """数据预处理：除以指标均值"""
    Mean = np.mean(X, axis=0)
    Z = X / Mean
    print('预处理后的矩阵为：')
    print(Z)
    return Z

def factor_correlation_analysis(Z):
    """因素关联分析：计算各指标与参考序列的关联度"""
    # 母序列（参考序列）
    Y = Z[:, 0]
    # 子序列（比较序列）
    X = Z[:, 1:]
    # 计算绝对差值矩阵
    absX0_Xi = np.abs(X - np.tile(Y.reshape(-1, 1), (1, X.shape[1])))
    # 计算两级最小差a
    a = np.min(absX0_Xi)
    # 计算两级最大差b
    b = np.max(absX0_Xi)
    # 分辨系数取0.5
    rho = 0.5
    # 计算关联系数
    gamma = (a + rho * b) / (absX0_Xi + rho * b)
    # 计算关联度（关联系数的平均值）
    grey_relational_degrees = np.mean(gamma, axis=0)
    
    print('\n--- 因素关联分析结果 ---')
    print('子序列中各指标与母序列的灰色关联度：')
    for i, degree in enumerate(grey_relational_degrees):
        print(f"指标{i+1}: {degree:.4f}")
    
    return grey_relational_degrees

def comprehensive_evaluation_analysis(Z):
    """综合评价分析：计算对象得分并排序"""
    # 构造虚拟母序列（每行最大值）
    Y = np.max(Z, axis=1)
    # 子序列即为预处理后的数据
    X = Z
    # 计算绝对差值矩阵
    absX0_Xi = np.abs(X - np.tile(Y.reshape(-1, 1), (1, X.shape[1])))
    # 计算两级最小差a
    a = np.min(absX0_Xi)
    # 计算两级最大差b
    b = np.max(absX0_Xi)
    # 分辨系数取0.5
    rho = 0.5
    # 计算关联系数
    gamma = (a + rho * b) / (absX0_Xi + rho * b)
    # 计算指标权重（关联度归一化）
    weight = np.mean(gamma, axis=0) / np.sum(np.mean(gamma, axis=0))
    # 计算未归一化得分
    score = np.sum(X * np.tile(weight, (X.shape[0], 1)), axis=1)
    # 归一化得分
    stand_S = score / np.sum(score)
    # 降序排序
    sorted_indices = np.argsort(stand_S)[::-1]
    sorted_scores = stand_S[sorted_indices]
    
    print('\n--- 综合评价分析结果 ---')
    print('指标权重：')
    for i, w in enumerate(weight):
        print(f"指标{i+1}: {w:.4f}")
    
    print('\n参评对象得分及排名：')
    print("排名 | 对象ID | 得分")
    for rank, idx in enumerate(sorted_indices):
        print(f" {rank+1:2d}  |   {idx+1:2d}    | {stand_S[idx]:.4f}")
    
    return stand_S, sorted_indices

if __name__ == "__main__":
    # 1. 输入数据
    n, m, kind, A = input_matrix()
    
    # 2. 统一指标类型
    X = unify_indicators(n, m, kind, A)
    
    # 3. 数据预处理
    Z = preprocess_matrix(X)
    
    # 4. 因素关联分析
    factor_correlation_analysis(Z.copy())
    
    # 5. 综合评价分析
    comprehensive_evaluation_analysis(Z.copy())