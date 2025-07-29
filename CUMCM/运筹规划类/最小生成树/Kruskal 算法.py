class UnionFind:
    """并查集类：用于检测环路和合并集合"""
    def __init__(self, n):
        # 初始化：每个节点的父节点指向自己，秩（树高）为0
        self.parent = list(range(n))  # 父节点数组
        self.rank = [0] * n           # 秩数组（用于优化合并）
    
    def find(self, x):
        """查找根节点并进行路径压缩"""
        if self.parent[x] != x:
            # 递归查找根节点，并将路径上的节点直接连到根节点
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]
    
    def union(self, x, y):
        """合并两个集合（按秩合并）"""
        root_x = self.find(x)
        root_y = self.find(y)
        if root_x == root_y:  # 已连通，无需合并
            return False
        
        # 按秩合并：小树合并到大树下
        if self.rank[root_x] < self.rank[root_y]:
            self.parent[root_x] = root_y
        elif self.rank[root_x] > self.rank[root_y]:
            self.parent[root_y] = root_x
        else:
            # 秩相等时，任意合并并增加秩
            self.parent[root_y] = root_x
            self.rank[root_x] += 1
        return True

def kruskal(n, edges):
    """
    Kruskal算法求解最小生成树
    :param n: 顶点数量
    :param edges: 边列表，格式 [(u, v, weight)]
    :return: 最小生成树的边列表
    """
    # 步骤1：初始化并查集和结果集
    uf = UnionFind(n)
    mst = []  # 存储最小生成树的边
    
    # 步骤2：按权重升序排序边 [1,9](@ref)
    edges.sort(key=lambda x: x[2])
    
    # 步骤3：遍历排序后的边
    for u, v, weight in edges:
        # 步骤4：检测是否形成环路
        if uf.find(u) != uf.find(v):
            # 步骤5：安全添加边（不会形成环）
            uf.union(u, v)
            mst.append((u, v, weight))
            # 优化：已收集n-1条边时提前终止
            if len(mst) == n - 1:
                break
    return mst

# 测试示例
if __name__ == "__main__":
    # 顶点数=6，边列表（格式：起点, 终点, 权重）
    edges = [
        (1, 2, 5), (1, 3, 6), 
        (4, 5, 7), (0, 1, 10), (1, 5, 11)
    ]
    mst = kruskal(6, edges)
    print("Kruskal算法结果：")
    for u, v, weight in mst:
        print(f"边 {u}-{v}, 权重: {weight}")