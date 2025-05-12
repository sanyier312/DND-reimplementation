import igraph as ig

# 定义函数查找最大强连通分量
def find_max_strong_component(g):
    # 查找强连通分量
    components = g.connected_components("STRONG")
    
    # 找到最大强连通分量的索引
    max_component_index = max(range(len(components)), key=lambda i: len(components[i]))
    
    # 获取最大强连通分量的顶点值
    max_component_vertices = components[max_component_index]
    
    # 返回最大强连通分量的顶点值和大小
    return {
        "vertices": max_component_vertices,
        "size": len(max_component_vertices)
    }

# 示例用法
# 创建一个有向图
g = ig.Graph(directed=True)
g.add_vertices(6)
g.add_edges([
        (1, 2), (2, 0),  # SCC1
        (3, 4), (4, 5), (5, 3),  # SCC2
        (2, 3)  # 桥接边
    ])
    

# 找到最大强连通分量
max_component = find_max_strong_component(g)

# 输出结果
print("最大强连通分量的顶点为：", max_component["vertices"])
print("最大强连通分量的大小为：", max_component["size"])