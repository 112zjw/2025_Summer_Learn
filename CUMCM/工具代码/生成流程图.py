from graphviz import Digraph

# 创建有向图，设置整体样式
dot = Digraph(comment='改进版SEIR模型', format='png')
dot.attr(rankdir='LR',  # 从左到右布局
         size='12,7',   # 图片大小
         dpi='300',
         concentrate='false')  # 避免边重叠

# 设置节点默认样式
dot.attr('node', shape='box',  # 节点形状
         style='filled,rounded',  # 填充和圆角
         fontname='SimHei',       # 支持中文
         fontsize='12',
         height='1.0',
         width='2.0')

# 设置边的默认样式
dot.attr('edge', fontname='SimHei',
         fontsize='10',
         penwidth='1.2')

# 添加节点（不同仓室使用不同颜色）
dot.node('S', '易感者\nS(t)', fillcolor='#a1caf1')
dot.node('E', '潜伏者\nE(t)', fillcolor='#b7f0b7')
dot.node('I', '现患者\nI(t)', fillcolor='#f4c2c2')
dot.node('R', '康复者\nR(t)', fillcolor='#fce5cd')
dot.node('N', '人口库\nN', fillcolor='#e0e0e0', style='filled,rounded,dashed')

# 添加主要传播路径（蓝色）
dot.edge('S', 'E', label='β·S·I/N', fontcolor='#1e78b4', color='#1e78b4', arrowhead='open')
dot.edge('E', 'I', label='σ·E', fontcolor='#1e78b4', color='#1e78b4', arrowhead='open')
dot.edge('I', 'R', label='γ·I', fontcolor='#1e78b4', color='#1e78b4', arrowhead='open')

# 添加免疫丧失路径（橙色）
dot.edge('R', 'S', label='ω·R', fontcolor='#ff7f00', color='#ff7f00', arrowhead='open')

# 添加人口动力学相关路径（紫色）
dot.edge('N', 'S', label='μ·N', fontcolor='#6a3d9a', color='#6a3d9a', arrowhead='open')
dot.edge('S', 'N', label='μ·S', fontcolor='#6a3d9a', color='#6a3d9a', style='dashed', arrowhead='open')
dot.edge('E', 'N', label='μ·E', fontcolor='#6a3d9a', color='#6a3d9a', style='dashed', arrowhead='open')
dot.edge('I', 'N', label='μ·I', fontcolor='#6a3d9a', color='#6a3d9a', style='dashed', arrowhead='open')
dot.edge('R', 'N', label='μ·R', fontcolor='#6a3d9a', color='#6a3d9a', style='dashed', arrowhead='open')

# 添加标题
dot.attr(label='改进版SEIR模型流程图（含出生率、死亡率和免疫有效性）', 
         fontname='SimHei', fontsize='14', fontcolor='#333333')
dot.attr(labelloc='t')  # 标题位置在上部

# 添加更清晰的图例（使用颜色块和文字说明）
with dot.subgraph(name='cluster_legend') as c:
    c.attr(label='路径说明', fontname='SimHei', style='dashed', margin='10')
    c.node('l1', '● 主要传播路径', shape='plaintext', fontcolor='#1e78b4')
    c.node('l2', '● 免疫丧失路径', shape='plaintext', fontcolor='#ff7f00')
    c.node('l3', '● 人口动力学（出生/死亡）', shape='plaintext', fontcolor='#6a3d9a')
    c.node('l4', '── 出生流程', shape='plaintext', fontcolor='#6a3d9a')
    c.node('l5', '--- 死亡流程', shape='plaintext', fontcolor='#6a3d9a')
    c.attr(rankdir='LR')
    c.edge('l1', 'l2', style='invis')  # 控制布局
    c.edge('l2', 'l3', style='invis')
    c.edge('l3', 'l4', style='invis')
    c.edge('l4', 'l5', style='invis')

# 调整节点布局，确保图例位置合理
dot.edge('R', 'cluster_legend', style='invis')

# 保存并显示图片
dot.render('improved_seir_model', view=True, cleanup=True)
    