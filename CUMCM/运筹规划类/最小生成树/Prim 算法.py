import heapq

def prim(n, edges):
    """
    Prim算法求解最小生成树
    :param n: 顶点数量
    :param edges: 边列表，格式 [(u, v, weight)]
    :return: 最小生成树的边列表
    """
    # 步骤1：构建邻接表
    graph = [[] for _ in range(n)]
    for u, v, weight in edges:
        graph[u].append((v, weight))
        graph[v].append((u, weight))  # 无向图需双向添加
    
    # 步骤2：初始化数据结构
    visited = [False] * n  # 顶点访问标记
    min_heap = []          # 最小堆存储候选边
    mst = []               # 结果集
    
    # 步骤3：从顶点0开始（可任选起点）
    start = 0
    visited[start] = True
    # 将起点的所有邻接边加入堆
    for neighbor, weight in graph[start]:
        heapq.heappush(min_heap, (weight, start, neighbor))
    
    # 步骤4：循环直到所有顶点连通
    while min_heap and len(mst) < n - 1:
        # 步骤5：弹出当前最小权重边
        weight, u, v = heapq.heappop(min_heap)
        if visited[v]:  # 跳过已访问顶点
            continue
        
        # 步骤6：添加安全边
        visited[v] = True
        mst.append((u, v, weight))
        
        # 步骤7：将新顶点的邻接边加入堆
        for neighbor, w in graph[v]:
            if not visited[neighbor]:
                heapq.heappush(min_heap, (w, v, neighbor))
    
    return mst

# 测试示例
if __name__ == "__main__":
    # 顶点数=6，边列表（格式：起点, 终点, 权重）
    edges = [
        (1, 2, 5), (1, 3, 6), 
        (4, 5, 7), (0, 1, 10), (1, 5, 11)
    ]
    mst = prim(6, edges)
    print("Prim算法结果：")
    for u, v, weight in mst:
        print(f"边 {u}-{v}, 权重: {weight}")