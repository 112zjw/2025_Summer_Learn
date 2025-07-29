
# 递归算法
# def coinChange(n):
#     if n == 0:
#         return 0
#     if n < 0:  # 处理无效金额
#         return float('inf')
    
#     res = float('inf')
#     # 并行尝试所有硬币面额（2、5、7）
#     if n >= 2:
#         res = min(res, coinChange(n - 2) + 1)
#     if n >= 5:
#         res = min(res, coinChange(n - 5) + 1)
#     if n >= 7:
#         res = min(res, coinChange(n - 7) + 1)
#     return res

#  动态规划算法
def coinChange(n):
    dp = [float('inf')] * (n + 1)  # 初始化动态规划数组，inf 表示初始时无法组成对应金额（inf为无限大）
    dp[0] = 0  # 找零金额为 0 时，需要 0 枚硬币
    for i in range(1, n + 1):
        if i >= 2:
            dp[i] = min(dp[i], dp[i - 2] + 1)
        if i >= 5:
            dp[i] = min(dp[i], dp[i - 5] + 1)
        if i >= 7:
            dp[i] = min(dp[i], dp[i - 7] + 1)
    if dp[n] != float('inf'):
        return dp[n]
    else:
        return -1

n = int(input('请输入要拼的金额: '))
res = coinChange(n)

print(res)