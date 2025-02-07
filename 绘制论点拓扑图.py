
import json
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
from matplotlib.lines import Line2D
import numpy as np

TOPIC = "向下的自由是不是自由"
FILEPATH = "cleaned_向下的自由是不是自由.json"

# 设置字体为 SimHei
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

def text_snippet(text, length=30):
    return text if len(text) <= length else text[:length] + "..."

def adjust_positions(pos, G, min_dist=0.15, min_dist_connected=0.05, iterations=100):
    """
    迭代调整布局：
      - 对于非直接连接的节点，确保距离不少于 min_dist；
      - 对于直接相连的节点（论点与其支撑/反驳），允许距离低至 min_dist_connected。
    """
    keys = list(pos.keys())
    pos_arr = {k: np.array(pos[k]) for k in keys}
    for it in range(iterations):
        for i in range(len(keys)):
            for j in range(i+1, len(keys)):
                k1 = keys[i]
                k2 = keys[j]
                p1 = pos_arr[k1]
                p2 = pos_arr[k2]
                delta_vec = p2 - p1
                dist = np.linalg.norm(delta_vec)
                # 若两节点直接相连，则使用较小阈值，否则使用较大阈值
                if G.has_edge(k1, k2) or G.has_edge(k2, k1):
                    threshold = min_dist_connected
                else:
                    threshold = min_dist
                if dist < threshold:
                    if dist == 0:
                        displacement = (np.random.rand(2) - 0.5) * threshold
                    else:
                        displacement = delta_vec / dist * (threshold - dist) / 2
                    pos_arr[k1] = p1 - displacement
                    pos_arr[k2] = p2 + displacement
    new_pos = {k: pos_arr[k] for k in keys}
    return new_pos

class UtteranceNode:
    def __init__(self, node_id, speaker, text, node_type, base_importance=0.0,
                 target_id=None, delta=0.0, round_number=None):
        """
        node_type:
          - "new_argument": 新增论点（需要 base_importance）
          - "support": 支持（需要 target_id 与正 delta）
          - "attack": 反驳（需要 target_id 与负 delta）
        """
        self.node_id = node_id
        self.speaker = speaker
        self.text = text
        self.node_type = node_type
        self.base_importance = base_importance
        self.target_id = target_id
        self.delta = delta
        self.round_number = round_number

    def to_dict(self):
        return {
            "id": self.node_id,
            "speaker": self.speaker,
            "text": self.text,
            "node_type": self.node_type,
            "base_importance": self.base_importance,
            "target_id": self.target_id,
            "delta": self.delta,
            "round_number": self.round_number
        }

class DebateGraph:
    def __init__(self):
        self.nodes = {}

    def add_node(self, node: UtteranceNode):
        self.nodes[node.node_id] = node

    def visualize_graph(self, filename="论点拓扑图.png", topic=""):
        # 构造有向图：只加入存在边连接的节点
        G = nx.DiGraph()
        nodes_with_edges = set()
        for node in self.nodes.values():
            if node.node_type in ("support", "attack") and node.target_id in self.nodes:
                nodes_with_edges.add(node.node_id)
                nodes_with_edges.add(node.target_id)
        for node_id in nodes_with_edges:
            node = self.nodes[node_id]
            
            ###### 这里可以根据需要修改节点标签的显示内容
            leng = min(30, len(node.text))
            label = f"{node.node_id[5:]}\n{text_snippet(node.text,length=30)[0:int(leng/2)]}\n{text_snippet(node.text)[int(leng/2):]}" # 显示节点 ID 和部分文本
            # label = f"{node.node_id[5:]}" # 仅显示节点 ID
            ############################################
            
            G.add_node(node.node_id, label=label, speaker=node.speaker, node_type=node.node_type)
        # 添加边时指定较高的权重，以增强聚类效果
        for node in self.nodes.values():
            if node.node_type in ("support", "attack") and node.target_id in self.nodes:
                G.add_edge(node.node_id, node.target_id, label=f"{node.delta}", node_type=node.node_type, weight=10)
        # 使用 spring_layout，利用边权参数将相关节点拉近
        pos = nx.spring_layout(G, weight='weight', seed=42)
        pos = adjust_positions(pos, G, min_dist=0.20, min_dist_connected=0.1, iterations=200)
        node_labels = nx.get_node_attributes(G, 'label')
        edge_labels = nx.get_edge_attributes(G, 'label')
        edge_labels = {edge: label for edge, label in edge_labels.items() if edge[0] in pos and edge[1] in pos}
        
        # 根据正反方和节点类型设置节点样式
        # 正方采用清新蓝色 (#64B5F6)，反方采用温暖橙红 (#E57373)
        # 节点形状：new_argument：圆形 "o"，support：正方形 "s"，attack：三角形 "^"
        node_groups = {}
        node_size_dict = {}
        for node_id in G.nodes:
            node = self.nodes[node_id]
            if node.speaker.lower() == "pro":
                color = "#64B5F6"
            elif node.speaker.lower() == "con":
                color = "#E57373"
            else:
                color = "gray"
            if node.node_type == "new_argument":
                marker = "o"
                size = abs(node.base_importance) * 1000 if node.base_importance != 0 else 300
            elif node.node_type == "support":
                marker = "s"
                size = abs(node.delta) * 1000 if node.delta != 0 else 300
            elif node.node_type == "attack":
                marker = "^"
                size = abs(node.delta) * 1000 if node.delta != 0 else 300
            else:
                marker = "o"
                size = 300
            key = (marker, color)
            if key not in node_groups:
                node_groups[key] = []
            node_groups[key].append(node_id)
            node_size_dict[node_id] = size

        plt.figure(figsize=(12, 8))
        # 分组绘制节点
        for (marker, color), nodelist in node_groups.items():
            sizes = [node_size_dict[nid] for nid in nodelist]
            pos_list = [pos[nid] for nid in nodelist]
            pos_array = np.array(pos_list)
            plt.scatter(pos_array[:, 0], pos_array[:, 1], s=sizes, c=color, marker=marker, edgecolors="k", zorder=3)
        # 绘制节点标签
        nx.draw_networkx_labels(G, pos, labels=node_labels, font_size=6)

        ax = plt.gca()
        # 绘制边和箭头
        for (u, v, d) in G.edges(data=True):
            if d['node_type'] == 'support':
                arrow = FancyArrowPatch(posA=pos[u], posB=pos[v], arrowstyle='->', color='green',
                                         mutation_scale=15, shrinkA=15, shrinkB=15)
            elif d['node_type'] == 'attack':
                arrow = FancyArrowPatch(posA=pos[u], posB=pos[v], arrowstyle='->', color='red',
                                         mutation_scale=15, shrinkA=15, shrinkB=15)
            ax.add_patch(arrow)
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_color='red',
                                     font_size=8, label_pos=0.5,
                                     bbox=dict(facecolor='white', edgecolor='none', alpha=0.6))
        plt.title(f"辩论论点图可视化\n{topic}")
        
        # 添加图例，说明正反方对应的颜色及节点形状对应的性质
        legend_items = [
            Line2D([0], [0], marker='o', color='w', label='正方论点', markerfacecolor="#64B5F6", markersize=10, markeredgecolor='k'),
            Line2D([0], [0], marker='s', color='w', label='正方支撑', markerfacecolor="#64B5F6", markersize=10, markeredgecolor='k'),
            Line2D([0], [0], marker='^', color='w', label='正方反驳', markerfacecolor="#64B5F6", markersize=10, markeredgecolor='k'),
            Line2D([0], [0], marker='o', color='w', label='反方论点', markerfacecolor="#E57373", markersize=10, markeredgecolor='k'),
            Line2D([0], [0], marker='s', color='w', label='反方支撑', markerfacecolor="#E57373", markersize=10, markeredgecolor='k'),
            Line2D([0], [0], marker='^', color='w', label='反方反驳', markerfacecolor="#E57373", markersize=10, markeredgecolor='k')
        ]
        plt.legend(handles=legend_items, loc='upper left', title="图例")
        
        plt.axis('off')
        plt.savefig(filename,dpi=300)
        plt.close()

def main(filepath=FILEPATH, topic=TOPIC):
    with open(filepath, 'r', encoding='utf-8') as f:
        nodes_list = json.load(f)
    graph = DebateGraph()
    for node_dict in nodes_list:
        node = UtteranceNode(
            node_id=node_dict["id"],
            speaker=node_dict["speaker"],
            text=node_dict["text"],
            node_type=node_dict["node_type"],
            base_importance=node_dict.get("base_importance", 0.0),
            target_id=node_dict.get("target_id"),
            delta=node_dict.get("delta", 0.0),
            round_number=node_dict.get("round_number")
        )
        graph.add_node(node)
    graph.visualize_graph(filename=filepath[0:10]+"论点拓扑图.png", topic=topic)
    print(f"\n论点拓扑图已保存为 {filepath[0:10]+"论点拓扑图.png"}.png")

if __name__ == "__main__":
    main()
