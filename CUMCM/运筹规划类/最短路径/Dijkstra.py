import networkx as nx
import matplotlib.pyplot as plt

# 解决matplotlib显示中文的问题
plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体
plt.rcParams['axes.unicode_minus'] = False    # 解决负号显示问题

# 定义图的边和权重
s = [9, 9, 1, 1, 3, 3, 3, 2, 2, 5, 5, 7, 7, 8]  # 起始节点
t = [1, 2, 2, 3, 4, 6, 7, 4, 5, 4, 7, 6, 8, 6]  # 终止节点
w = [4, 8, 3, 8, 2, 7, 4, 1, 6, 6, 2, 14, 10, 9]  # 边的权重

# ==================== 创建带权图 ====================
G = nx.Graph()
# 添加带权重的边
for i in range(len(s)):
    G.add_edge(s[i], t[i], weight=w[i])

# ==================== 绘制原始图 ====================
plt.figure(figsize=(10, 8))
plt.title("原始带权图")
pos = nx.spring_layout(G, seed=42)  # 固定布局以保证一致性
labels = nx.get_edge_attributes(G, 'weight')
nx.draw(G, pos, with_labels=True, node_size=800, 
        node_color='#A0CBE2', font_size=12, font_weight='bold',
        edge_color='gray', width=2)
nx.draw_networkx_edge_labels(G, pos, edge_labels=labels, font_size=10)
plt.tight_layout()
plt.savefig("原始图.png", dpi=300)
plt.show()

# ==================== 计算最短路径 ====================
# 计算从节点9到节点8的最短路径
P = nx.shortest_path(G, source=9, target=8, weight='weight')
d = nx.shortest_path_length(G, source=9, target=8, weight='weight')
print("\n" + "="*50)
print(f"从节点 9 到节点 8 的最短路径: {P}")
print(f"最短路径总权重: {d}")
print("="*50 + "\n")

# ==================== 高亮显示最短路径 ====================
plt.figure(figsize=(10, 8))
plt.title("最短路径高亮 (9 → 8)")
nx.draw(G, pos, with_labels=True, node_size=800, 
        node_color='#A0CBE2', font_size=12, font_weight='bold',
        edge_color='lightgray', width=2)

# 高亮最短路径
path_edges = list(zip(P[:-1], P[1:]))
nx.draw_networkx_edges(G, pos, edgelist=path_edges, 
                       edge_color='red', width=4)
nx.draw_networkx_nodes(G, pos, nodelist=P, 
                       node_color='#FF7F0E', node_size=900)  # 路径节点高亮
nx.draw_networkx_edge_labels(G, pos, edge_labels=labels, font_size=10)

# 添加路径信息标注
plt.annotate(f"最短路径长度: {d}", xy=(0.02, 0.02), 
             xycoords='figure fraction', fontsize=12)
plt.tight_layout()
plt.savefig("最短路径图.png", dpi=300)
plt.show()

# ==================== 计算全图最短路径距离 ====================
D = dict(nx.all_pairs_dijkstra_path_length(G, weight='weight'))

# 格式化输出距离矩阵
print("\n" + "="*50)
print("最短路径距离矩阵:")
nodes = sorted(G.nodes())
print(f"{'节点':<5}", end="")
for n in nodes:
    print(f"{n:<5}", end="")
print("\n" + "-"*50)

for i in nodes:
    print(f"{i:<5}", end="")
    for j in nodes:
        print(f"{D[i].get(j, '∞'):<5}", end="")
    print()
print("="*50 + "\n")

# ==================== 特定节点间距离查询 ====================
print("[节点间最短距离查询]")
print(f"节点 1 → 2 的最短距离: {D[1][2]}")
print(f"节点 9 → 4 的最短距离: {D[9][4]}\n")

# ==================== 查找临近节点 ====================
nearest_nodes = nx.single_source_dijkstra_path_length(G, 2, cutoff=10, weight='weight')
print("节点2的邻居节点（距离≤10）:")
print("{:<8} {:<10}".format("节点", "距离"))
print("-"*20)
for node, dist in nearest_nodes.items():
    print("{:<10} {:<10.2f}".format(node, dist))