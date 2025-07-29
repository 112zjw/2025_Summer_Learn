import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

# ====================== 中文字体配置 ======================
# 设置支持中文的字体（黑体）
plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体显示中文
plt.rcParams['axes.unicode_minus'] = False    # 解决负号显示问题[7,9](@ref)

# ====================== 1. 初始化画布与标题 ======================
# 创建2行3列的子图布局，整体画布尺寸18x12英寸
fig, axes = plt.subplots(2, 3, figsize=(18, 12))
# 添加中文总标题，字体大小20
fig.suptitle('NetworkX图可视化示例', fontsize=20)

# ====================== 2. 图1：未加权无向图（整数节点） ======================
s1 = [1, 2, 3, 4]
t1 = [2, 3, 1, 1]
G1 = nx.Graph()
G1.add_edges_from(zip(s1, t1))  # 通过边列表构建图
# 在子图位置[0,0]绘制，设置节点颜色、大小和标签
nx.draw(G1, ax=axes[0, 0], with_labels=True, node_color='lightblue', 
        node_size=500, edge_color='gray', font_size=12)
axes[0, 0].set_title('1. 未加权图（整数节点）', fontsize=14)  # 中文标题
axes[0, 0].set_xticks([])  # 隐藏坐标轴刻度
axes[0, 0].set_yticks([])

# ====================== 3. 图2：未加权无向图（字符串节点） ======================
s2 = ['School', 'Cinema', 'Mall', 'Hotel']
t2 = ['Cinema', 'Hotel', 'Hotel', 'KTV']
G2 = nx.Graph()
G2.add_edges_from(zip(s2, t2))
# 在子图位置[0,1]绘制，节点颜色设为浅绿色
nx.draw(G2, ax=axes[0, 1], with_labels=True, node_color='lightgreen', 
        node_size=500, edge_color='gray', font_size=12, width=2)
axes[0, 1].set_title('2. 未加权图（字符串节点）', fontsize=14)  # 中文标题
axes[0, 1].set_xticks([])
axes[0, 1].set_yticks([])

# ====================== 4. 图3：加权无向图（手动添加权重） ======================
s = [1, 2, 3, 4]
t = [2, 3, 1, 1]
w = [3, 8, 9, 2]
G3 = nx.Graph()
for i in range(len(s)):
    G3.add_edge(s[i], t[i], weight=w[i])  # 为每条边添加权重属性
    
pos = nx.spring_layout(G3, seed=42)  # 固定布局种子确保可重现性
nx.draw(G3, pos, ax=axes[0, 2], with_labels=True, node_color='lightblue', 
        node_size=500, edge_color='gray', font_size=12, width=2)
labels = nx.get_edge_attributes(G3, 'weight')  # 提取权重属性
nx.draw_networkx_edge_labels(G3, pos, ax=axes[0, 2], edge_labels=labels)  # 显示权重标签
axes[0, 2].set_title('3. 加权无向图', fontsize=14)  # 中文标题
axes[0, 2].set_xticks([])
axes[0, 2].set_yticks([])

# ====================== 5. 图4：邻接矩阵构建的加权图 ======================
a = [[0, 3, 9, 2],
     [3, 0, 8, 0],
     [9, 8, 0, 0],
     [2, 0, 0, 0]]
G4 = nx.from_numpy_array(np.array(a))   # 直接从邻接矩阵创建图
pos = nx.spring_layout(G4, seed=42)  # 保持与图3相同的布局种子
nx.draw(G4, pos, ax=axes[1, 0], with_labels=True, node_color='lightblue', 
        node_size=500, edge_color='gray', font_size=12, width=2)
labels = nx.get_edge_attributes(G4, 'weight')
nx.draw_networkx_edge_labels(G4, pos, ax=axes[1, 0], edge_labels=labels)
axes[1, 0].set_title('4. 邻接矩阵加权图', fontsize=14)  # 中文标题
axes[1, 0].set_xticks([])
axes[1, 0].set_yticks([])

# ====================== 6. 图5：加权有向图 ======================
s = [1, 2, 3, 4]
t = [2, 3, 1, 1]
w = [3, 8, 9, 2]
G5 = nx.DiGraph()  # 创建有向图对象
for i in range(len(s)):
    G5.add_edge(s[i], t[i], weight=w[i])

pos = nx.spring_layout(G5, seed=42)
# 有向图需设置arrows=True，节点颜色区分
nx.draw(G5, pos, ax=axes[1, 1], with_labels=True, node_color='salmon', 
        node_size=500, edge_color='gray', font_size=12, width=2, 
        arrows=True, arrowsize=20)  # arrowsize控制箭头大小
labels = nx.get_edge_attributes(G5, 'weight')
nx.draw_networkx_edge_labels(G5, pos, ax=axes[1, 1], edge_labels=labels)
axes[1, 1].set_title('5. 加权有向图', fontsize=14)  # 中文标题
axes[1, 1].set_xticks([])
axes[1, 1].set_yticks([])

# ====================== 7. 图6：邻接矩阵构建的有向图 ======================
a = [[0, 3, 0, 0],
     [3, 0, 8, 0],
     [9, 0, 0, 0],
     [2, 0, 0, 0]]
# create_using指定图类型为有向图
G6 = nx.from_numpy_array(np.array(a), create_using=nx.DiGraph)
pos = nx.spring_layout(G6, seed=42)
nx.draw(G6, pos, ax=axes[1, 2], with_labels=True, node_color='salmon', 
        node_size=500, edge_color='gray', font_size=12, width=2, 
        arrows=True, arrowsize=20)
labels = nx.get_edge_attributes(G6, 'weight')
nx.draw_networkx_edge_labels(G6, pos, ax=axes[1, 2], edge_labels=labels)
axes[1, 2].set_title('6. 邻接矩阵有向图', fontsize=14)  # 中文标题
axes[1, 2].set_xticks([])
axes[1, 2].set_yticks([])

# ====================== 8. 布局优化 ======================
plt.tight_layout(rect=[0, 0, 1, 0.96])  # 调整子图间距，为总标题预留空间
plt.subplots_adjust(wspace=0.3, hspace=0.2)  # 设置水平/垂直间距
plt.show()