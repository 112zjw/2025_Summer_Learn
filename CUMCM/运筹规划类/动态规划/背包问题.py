def knapsack(n, W, weights, values):
    """
    使用动态规划解决0-1背包问题
    
    参数:
    n (int): 物品数量
    W (int): 背包最大容量
    weights (list): 每个物品的重量列表
    values (list): 每个物品的价值列表
    
    返回:
    int: 背包能装载物品的最大价值
    """
    # 初始化二维DP数组，大小为(n+1) x (W+1)
    # dp[i][j] 表示前i件物品放入容量为j的背包中所获得的最大价值
    dp = [[0] * (W + 1) for i in range(n + 1)]
    
    # 动态规划求解
    for i in range(1, n + 1):      # 遍历物品（从1到n）
        for j in range(1, W + 1):  # 遍历背包容量（从1到W）
            # 如果当前物品的重量大于背包容量j，则无法放入当前物品
            if weights[i-1] > j:
                # 最大价值等于前i-1件物品放入容量j的背包的最大价值
                dp[i][j] = dp[i-1][j]
            else:
                # 可以选择放入或不放入当前物品，取价值更大的方案：
                # 不放：dp[i-1][j]（保留前i-1件物品的最大价值）
                # 放入：dp[i-1][j-weights[i-1]] + values[i-1] 
                #       (前i-1件物品在剩余空间 j-weight[i-1] 的最大价值 + 当前物品价值)
                dp[i][j] = max(dp[i-1][j], dp[i-1][j-weights[i-1]] + values[i-1])
    
    # 最终结果：dp[n][W]表示所有物品放入容量W背包的最大价值
    return dp[n][W]

# 测试数据（根据您图片中的示例）
if __name__ == "__main__":
    # 物品数量
    n = 3
    # 背包容量
    W = 5
    # 物品重量列表
    weights = [1,2,3]
    # 物品价值列表
    values = [3,4,5]
    
    # 计算并输出最大价值
    max_value = knapsack(n, W, weights, values)
    print(f"背包能装载物品的最大价值为: {max_value}")