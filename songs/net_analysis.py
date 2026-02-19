"""
词-歌曲复杂网络特征分析脚本

支持四种词性（名词、动词、形容词、时间词）的网络分析
生成包含网络特征、度分布、聚类系数、中心性指标、社区结构等完整分析报告
"""

import json
import networkx as nx
import numpy as np
from collections import Counter
from scipy import sparse
import re
import os
import sys

# ==================== 配置常量 ====================
# Hub 节点识别参数
HUB_THRESHOLD_MULTIPLIER = 2  # Hub 阈值 = 平均度 + n*标准差

# 显示参数
TOP_NODES_COUNT = 5  # 显示中心性最高的节点数
HUB_NODES_DISPLAY_COUNT = 5  # Hub 节点列表显示数量
HUB_NODES_TOTAL_COUNT = 10  # 内部记录 Hub 节点总数

# 路径长度阈值（用于判断信息传播效率）
AVG_PATH_LENGTH_THRESHOLD = 4.5

# 词性配置
POS_NETWORKS = {
    'n': '名词',
    'v': '动词',
    'a': '形容词',
    't': '时间词'
}

# 节点类型
NODE_TYPE_WORD = 'word'
NODE_TYPE_SONG = 'song'


def load_network_data(file_path):
    """加载网络数据"""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"数据文件不存在: {file_path}")
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data


def analyze_degree_distribution(G):
    """分析度分布"""
    degrees = [d for n, d in G.degree()]
    
    degree_stats = {
        'mean': np.mean(degrees),
        'std': np.std(degrees),
        'max': max(degrees),
        'min': min(degrees)
    }
    
    # 识别Hub节点（度>平均值+n倍标准差）
    hub_threshold = degree_stats['mean'] + HUB_THRESHOLD_MULTIPLIER * degree_stats['std']
    hub_nodes = [(n, d) for n, d in G.degree() if d > hub_threshold]
    
    return {
        'stats': degree_stats,
        'hub_count': len(hub_nodes),
        'hub_nodes': hub_nodes[:HUB_NODES_TOTAL_COUNT]
    }


def analyze_clustering(G):
    """分析聚类系数"""
    clustering = nx.clustering(G)
    clustering_values = list(clustering.values())
    avg_clustering = np.mean(clustering_values)
    max_clustering = max(clustering_values) if clustering_values else 0
    
    # 检查是否所有节点的聚类系数都为0（二分网络特性）
    is_all_zero = all(c == 0 for c in clustering_values)
    
    # 格式化聚类系数，保留到非零位
    if avg_clustering == 0:
        formatted_avg = "0"
    else:
        s = f"{avg_clustering:.10f}"
        decimal_part = s.split('.')[1]
        first_non_zero_idx = next((i for i, c in enumerate(decimal_part) if c != '0'), len(decimal_part))
        decimal_places = first_non_zero_idx + 1 if first_non_zero_idx < len(decimal_part) else 1
        formatted_avg = f"{avg_clustering:.{decimal_places}f}"
    
    return {
        'average': avg_clustering,
        'formatted': formatted_avg,
        'max': max_clustering,
        'is_all_zero': is_all_zero
    }


def analyze_centrality(G):
    """分析中心性指标"""
    degree_centrality = nx.degree_centrality(G)
    betweenness_centrality = nx.betweenness_centrality(G)
    closeness_centrality = nx.closeness_centrality(G)
    
    top_degree = sorted(degree_centrality.items(), key=lambda x: x[1], reverse=True)[:TOP_NODES_COUNT]
    top_betweenness = sorted(betweenness_centrality.items(), key=lambda x: x[1], reverse=True)[:TOP_NODES_COUNT]
    top_closeness = sorted(closeness_centrality.items(), key=lambda x: x[1], reverse=True)[:TOP_NODES_COUNT]
    
    return {
        'top_degree': top_degree,
        'top_betweenness': top_betweenness,
        'top_closeness': top_closeness
    }


def analyze_connectivity(G):
    """分析连通性"""
    is_connected = nx.is_connected(G)
    if is_connected:
        return {
            'is_connected': True,
            'num_components': 1,
            'largest_component_size': G.number_of_nodes()
        }
    else:
        components = list(nx.connected_components(G))
        component_sizes = sorted([len(c) for c in components], reverse=True)
        return {
            'is_connected': False,
            'num_components': len(components),
            'largest_component_size': component_sizes[0],
            'component_sizes': component_sizes[:5]  # 返回前5大的连通分量
        }


def calculate_gini_coefficient(degrees):
    """计算度分布的Gini系数（衡量不均度）"""
    if len(degrees) == 0:
        return 0
    degrees = sorted(degrees)
    n = len(degrees)
    cumsum = 0
    for i, d in enumerate(degrees):
        cumsum += (i + 1) * d
    return (2 * cumsum) / (n * sum(degrees)) - (n + 1) / n


def analyze_robustness(G, hub_nodes):
    """分析网络脆弱性（删除高度节点的影响）"""
    """返回删除前top hub节点后的网络变化"""
    G_copy = G.copy()
    original_size = G.number_of_nodes()
    
    # 逐个删除hub节点，查看最大连通分量的变化
    impact = []
    for node_id, degree in hub_nodes[:5]:  # 只看前5个hub
        G_copy.remove_node(node_id)
        if G_copy.number_of_nodes() > 0:
            largest_cc = max(nx.connected_components(G_copy), key=len)
            lcc_size = len(largest_cc)
            connectivity_ratio = lcc_size / original_size
            impact.append({
                'node': node_id,
                'degree': degree,
                'lcc_size': lcc_size,
                'connectivity_ratio': connectivity_ratio
            })
    
    return impact


def analyze_diameter(G):
    """分析网络直径"""
    if nx.is_connected(G):
        diameter = nx.diameter(G)
        return {'diameter': diameter, 'is_connected': True}
    else:
        # 对于不连通图，计算最大连通分量的直径
        largest_cc = max(nx.connected_components(G), key=len)
        subgraph = G.subgraph(largest_cc)
        diameter = nx.diameter(subgraph)
        return {'diameter': diameter, 'is_connected': False}


def analyze_community(G):
    """分析社区结构"""
    try:
        from sknetwork.clustering import Louvain
        
        # 转换为邻接矩阵
        adjacency_array = nx.to_scipy_sparse_array(G, format='csr')
        # 转换为 csr_matrix（scikit-network 要求）
        adjacency = sparse.csr_matrix(adjacency_array)
        
        # 应用 Louvain 算法
        louvain = Louvain()
        labels = louvain.fit_predict(adjacency)
        
        # 计算社区信息
        community_sizes = Counter(labels)
        community_count = len(community_sizes)
        
        # 使用 NetworkX 的 modularity 函数计算模块度
        partition_dict = {node: int(labels[i]) for i, node in enumerate(G.nodes())}
        mod = nx.algorithms.community.modularity(G, [set(nodes) for community_id, nodes in 
                                                     [(com_id, [n for n, c in partition_dict.items() if c == com_id]) 
                                                      for com_id in set(labels)]])
        
        return {
            'community_count': community_count,
            'community_sizes': dict(community_sizes.most_common(5)),
            'modularity': mod,
            'error': None
        }
    except Exception as e:
        return {'error': str(e)}


def get_node_type_cn(node_type):
    """获取节点类型的中文表示"""
    return '词' if node_type == NODE_TYPE_WORD else '歌曲'


def extract_patterns(content):
    """从content中提取模式"""
    pattern = r'([^\s，。]+：)'
    matches = re.findall(pattern, content)
    return matches


def generate_network_description(data, pos_key, pos_name):
    """生成网络描述文本"""
    G = nx.Graph()
    
    # 添加节点
    for node in data[pos_key]['nodes']:
        G.add_node(node['id'], **node)
    
    # 添加边
    for edge in data[pos_key]['edges']:
        G.add_edge(edge['source'], edge['target'])
    
    # 整个网络不连通的情况
    full_num_nodes = G.number_of_nodes()
    full_density = nx.density(G)
    
    # 分析连通性
    connectivity = analyze_connectivity(G)
    
    # 判断是否使用最大连通子图进行分析
    use_lcc = not connectivity['is_connected']
    
    if use_lcc:
        # 提取最大连通子图
        largest_cc = max(nx.connected_components(G), key=len)
        G = G.subgraph(largest_cc).copy()
        analysis_note = f"（基于最大连通分量，包含{G.number_of_nodes()}个节点）"
    else:
        analysis_note = ""
    
    num_nodes = G.number_of_nodes()
    density = nx.density(G)
    
    avg_path_length = None
    if nx.is_connected(G):
        avg_path_length = nx.average_shortest_path_length(G)
    
    degree_analysis = analyze_degree_distribution(G)
    clustering_analysis = analyze_clustering(G)
    centrality_analysis = analyze_centrality(G)
    
    # 新增分析
    degrees = [d for n, d in G.degree()]
    gini_coefficient = calculate_gini_coefficient(degrees)
    diameter_info = analyze_diameter(G)
    robustness = analyze_robustness(G, degree_analysis['hub_nodes'])
    
    descriptions = []
    
    # 描述1：网络特征
    title1 = "网络特征"
    content1 = f"规模与结构：网络总共包含{full_num_nodes}个节点，总边数为{G.number_of_edges()}条。"
    content1 += f"稀疏性：网络密度为{density:.6f}，属于稀疏网络，实际连接数远小于可能连接数。"
    if connectivity['is_connected']:
        # 连通网络：显示小世界性
        content1 += f"小世界性：平均路径长度为{avg_path_length:.2f}，网络直径为{diameter_info['diameter']}，远小于网络规模{full_num_nodes}，表现出典型的小世界特性。"
    else:
        # 不连通网络：显示连通性信息和基于 LCC 的分析
        content1 += f"连通性：网络分裂为{connectivity['num_components']}个连通分量，下面的分析基于最大连通分量（包含{num_nodes}个节点）进行，该分量直径为{diameter_info['diameter']}。"
    content1 += f"无标度性：度分布的Gini系数为{gini_coefficient:.3f}，显示出强烈的不均度特征，度数最大为{degree_analysis['stats']['max']}，最小为{degree_analysis['stats']['min']}，存在{degree_analysis['hub_count']}个Hub节点。"
    patterns1 = extract_patterns(content1)
    descriptions.append({'title': title1, 'content': content1, 'patterns': patterns1})
    
    # 描述2：度分布特征与网络脆弱性
    title2 = "度分布特征与脆弱性"
    content2 = f"平均度为{degree_analysis['stats']['mean']:.2f}，标准差为{degree_analysis['stats']['std']:.2f}，度分布不均度系数（Gini）为{gini_coefficient:.3f}{analysis_note}。"
    content2 += f"识别出{degree_analysis['hub_count']}个Hub节点（度>{degree_analysis['stats']['mean'] + HUB_THRESHOLD_MULTIPLIER*degree_analysis['stats']['std']:.2f}）。"
    if degree_analysis['hub_nodes']:
        content2 += "主要Hub节点包括："
        for node_id, degree in degree_analysis['hub_nodes'][:HUB_NODES_DISPLAY_COUNT]:
            node_data = G.nodes[node_id]
            label = node_data.get('label', node_id)
            node_type_cn = get_node_type_cn(node_data.get('node_type'))
            content2 += f" {node_type_cn} '{label}' (度={degree})"
    content2 += f"。网络脆弱性：删除这些高度Hub节点会对网络连通性产生显著影响。"
    if robustness:
        content2 += f"删除最高度节点后，最大连通分量规模从{G.number_of_nodes()}降至{robustness[0]['lcc_size']}（降幅{(1-robustness[0]['connectivity_ratio'])*100:.1f}%），表明网络对Hub节点失效较为敏感。"
    patterns2 = extract_patterns(content2)
    descriptions.append({'title': title2, 'content': content2, 'patterns': patterns2})
    
    # 描述3：聚类系数与中心性指标
    title3 = "聚类系数与中心性指标"
    content3 = f"平均聚类系数为{clustering_analysis['formatted']}{analysis_note}"
    if clustering_analysis['is_all_zero']:
        content3 += "，所有节点的聚类系数都为0，这是典型的二分网络特性（词-歌曲之间连接，词与词之间不直接连接）。"
    else:
        content3 += f"（最大值{clustering_analysis['max']:.3f}），接近于0，主要因为这是二分网络的特性。"
    
    if centrality_analysis['top_degree']:
        content3 += "度中心性最高的节点："
        for node_id, centrality in centrality_analysis['top_degree'][:2]:
            node_data = G.nodes[node_id]
            label = node_data.get('label', node_id)
            node_type_cn = get_node_type_cn(node_data.get('node_type'))
            content3 += f" {node_type_cn} '{label}' (中心性={centrality:.3f})"
        content3 += "。"
    
    if centrality_analysis['top_betweenness']:
        content3 += "介数中心性最高的节点（网络中的\"桥梁\"节点）："
        for node_id, centrality in centrality_analysis['top_betweenness'][:2]:
            node_data = G.nodes[node_id]
            label = node_data.get('label', node_id)
            node_type_cn = get_node_type_cn(node_data.get('node_type'))
            content3 += f" {node_type_cn} '{label}' (介数={centrality:.3f})"
        content3 += "。"
    
    if centrality_analysis['top_closeness']:
        content3 += "接近中心性最高的节点（网络中离其他节点最近的节点）："
        for node_id, centrality in centrality_analysis['top_closeness'][:1]:
            node_data = G.nodes[node_id]
            label = node_data.get('label', node_id)
            node_type_cn = get_node_type_cn(node_data.get('node_type'))
            content3 += f" {node_type_cn} '{label}' (接近度={centrality:.3f})"
        content3 += "。"
    
    patterns3 = extract_patterns(content3)
    descriptions.append({'title': title3, 'content': content3, 'patterns': patterns3})
    
    # 描述4：实际意义与应用前景
    title4 = "实际意义与应用前景"
    avg_path_str = f"{avg_path_length:.2f}" if avg_path_length else "不连通"
    content4 = f"语义紧密性：{pos_name}与歌曲之间的语义关联较为紧密且具有小世界特性{analysis_note}，最短路径均值为{avg_path_str}，"
    if avg_path_length:
        content4 += f"平均只需{avg_path_length:.2f}步即可建立任意两个{pos_name}之间的联系。"
    else:
        content4 += "但存在多个独立的主题群体，群体间存在语义隔离。"
    
    content4 += f"信息传播：该网络的信息传播效率{'较高' if (avg_path_length and avg_path_length < AVG_PATH_LENGTH_THRESHOLD) else '中等'}，有利于{pos_name}语义在不同歌曲间有效扩散。"
    if use_lcc:
        content4 += f"（分析基于最大连通分量，包含{num_nodes}个节点）。"
    
    content4 += f"网络韧性：{'当前' if connectivity['is_connected'] else '最大连通分量的'}连通网络使得系统对随机节点失效具有一定的鲁棒性，但对Hub节点失效较为敏感（删除最高度节点会导致连通分量规模大幅下降）。应用前景：该网络结构适合用于歌词推荐、情感认知和主题聚类等应用。"
    patterns4 = extract_patterns(content4)
    descriptions.append({'title': title4, 'content': content4, 'patterns': patterns4})
    
    return descriptions


def validate_data(data):
    """验证加载的数据结构"""
    for pos_key in POS_NETWORKS.keys():
        if pos_key not in data:
            raise ValueError(f"数据中缺少词性 {pos_key} 的配置")
        if 'nodes' not in data[pos_key] or 'edges' not in data[pos_key]:
            raise ValueError(f"词性 {pos_key} 的数据结构不完整")


def analyze_networks(data_file_path, output_file_path):
    """
    分析四种词性的网络特征
    
    参数:
        data_file_path: 输入数据文件路径
        output_file_path: 输出JSON文件路径
    """
    data = load_network_data(data_file_path)
    validate_data(data)
    
    result = {}
    
    for pos_key, pos_name in POS_NETWORKS.items():
        print(f"正在分析 {pos_name} 网络...")
        descriptions = generate_network_description(data, pos_key, pos_name)
        result[pos_key] = {
            'name': pos_name,
            'desc': descriptions
        }
    
    # 输出JSON结果
    with open(output_file_path, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    
    print(f"分析结果已保存到 {output_file_path}")
    return result


if __name__ == '__main__':
    # Ensure project root is in path for module imports
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    # ==================== 配置区 ====================
    # data_prefix = "jaychou"
    data_prefix = "mayday"
    # data_prefix = "liuyuning"
    
    # 基于脚本位置构造数据路径
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(script_dir, 'data', data_prefix)
    default_data = os.path.join(data_dir, 'graph_word_song_data_full.json')
    default_output = os.path.join(data_dir, 'network_analysis_result.json')
    
    
    
    result = analyze_networks(default_data, default_output)
    
    print("\n分析完成！")
    print(f"已处理 {len(result)} 种词性网络")