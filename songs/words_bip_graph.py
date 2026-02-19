# -*- encoding: utf-8 -*-
'''
@File    : words_bip_graph.py
@Time    : 2026/02/15
@Description : 歌词网络图数据生成器 (解耦重构版)
'''

import pandas as pd
import json
import networkx as nx
import numpy as np
import math
from collections import defaultdict
import sys
import os

# Ensure project root is in path for module imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from songs.songs_libs import id_process, LyricProcessConstants
except ImportError:
    try:
        from songs_libs import id_process, LyricProcessConstants
    except ImportError:
        print("警告: 无法从 'songs.songs_libs' 或 'songs_libs' 导入必要的模块。")
        class LyricProcessConstants:
            STOP_WORDS_V = []
            STOP_WORDS_N = []
            STOP_WORDS_A = []
            STOP_WORDS_T = []

# --- 1. 数据加载与基础处理层 ---

class LyricsDataLoader:
    """负责数据的读取与基础属性配置"""
    def __init__(self, data_path, is_ost=False):
        self.data_path = data_path
        self.is_ost = is_ost
        self.df_raw = None
        self.stop_words_v = LyricProcessConstants.STOP_WORDS_V
        self.stop_words_n = LyricProcessConstants.STOP_WORDS_N
        self.stop_words_a = LyricProcessConstants.STOP_WORDS_A
        self.stop_words_t = LyricProcessConstants.STOP_WORDS_T

        # 自动推断专辑类型描述
        self.album_type_desc = '录音室专辑'
        if "mayday" in data_path:
            self.album_type_desc = '录音室&精选辑'
        elif "liuyuning" in data_path:
            self.album_type_desc = '影视剧OST歌曲'

    def load(self):
        """惰性加载数据"""
        if self.df_raw is None:
            file_path = os.path.join(self.data_path, "cleared_words_data.csv")
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"找不到数据文件: {file_path}")
            self.df_raw = pd.read_csv(file_path)
            print(f"数据加载成功，共 {len(self.df_raw)} 行。")
        return self.df_raw

    def get_total_counts(self):
        """获取总歌曲数和专辑数"""
        df = self.load()
        return {
            'song_count': df['song_id'].nunique(),
            'album_count': df['album_id'].nunique()
        }

# --- 2. 业务逻辑处理层 ---

class MetricsCalculator:
    """负责核心指标计算与子集筛选"""
    def __init__(self, data_loader):
        self.loader = data_loader
        self.stop_words_v = data_loader.stop_words_v
        self.stop_words_n = data_loader.stop_words_n
        self.stop_words_a = data_loader.stop_words_a
        self.stop_words_t = data_loader.stop_words_t
        # 停用词字典映射，便于统一管理
        self.stop_words_dict = {
            'v': self.stop_words_v,
            'n': self.stop_words_n,
            'a': self.stop_words_a,
            't': self.stop_words_t
        }

    def _filter_stop_words(self, df, pos):
        """通用停用词过滤方法"""
        stop_words = self.stop_words_dict.get(pos, [])
        if stop_words:
            return df[~df['word'].isin(stop_words)]
        return df

    def _count_word_stats(self, df, pos, words_num=None, is_starts_with=True, filter_stopwords=True):
        """统计词频和歌曲覆盖数"""
        # 可选：过滤停用词
        df_working = self._filter_stop_words(df, pos) if filter_stopwords else df

        if is_starts_with:
            df_pos = df_working[df_working['pos'].str.startswith(pos, na=False)]
        else:
            df_pos = df_working[df_working['pos'] == pos]

        if df_pos.empty:
            return pd.DataFrame(columns=['word', 'song_num', 'freq', 'order'])

        # 统计覆盖歌曲数 (Document Frequency)
        word_cnt = df_pos.groupby('word')['song_id'].nunique().reset_index().rename(columns={'song_id': 'song_num'})
        # 统计词频 (Term Frequency)
        word_sum = df_pos.groupby('word')['freq'].sum().reset_index()

        # 合并
        if words_num and words_num > 0:
            word_cnt = word_cnt.sort_values(by='song_num', ascending=False)
            res = word_cnt.head(words_num).merge(word_sum, on='word', how='left')
        else:
            # 不限制数量
            res = word_cnt.merge(word_sum, on='word', how='left')

        res['order'] = 100 - res.index
        return res

    def get_subset(self, pos, words_num=None, is_starts_with=True):
        """获取用于绘图的清洗后子集"""
        df_raw = self.loader.load()
        df_working = df_raw.copy()

        # 1. 过滤停用词 (使用统一的方法)
        df_working = self._filter_stop_words(df_working, pos)

        # 2. 获取目标高频词
        stats = self._count_word_stats(df_working, pos, words_num, is_starts_with, filter_stopwords=False)
        target_words = set(stats['word'])

        # 3. 筛选行
        if is_starts_with:
            mask = df_working['pos'].str.startswith(pos, na=False)
        else:
            mask = df_working['pos'] == pos

        df_subset = df_working[mask].copy()
        df_subset = df_subset[df_subset['word'].isin(target_words)]

        # 4. 统一处理
        df_subset['pos'] = pos
        if id_process:
            df_subset = id_process(df_subset, is_ost=self.loader.is_ost)

        if 'album_order' in df_subset.columns:
            df_subset = df_subset.sort_values(by=['album_order'])

        return df_subset.reset_index(drop=True)

# --- 3. 布局引擎层 ---

class LayoutEngine:
    """负责节点坐标计算"""

    @staticmethod
    def calculate_embracing_layout(nodes, stretch_factor=None):
        """双扇形布局算法"""
        max_degree = 0
        if nodes:
            max_degree = max(nodes, key=lambda x: x['degree'])['degree']

        if stretch_factor is None:
            thresholds = [(80, 0.76), (70, 0.79), (60, 0.82), (50, 0.85)]
            stretch_factor = next((val for thresh, val in thresholds if max_degree > thresh), 0.85)

        coords = {}
        arc_radius = 800
        arc_span = math.pi / 1.6
        arc_total_height = 2 * (arc_radius * math.sin(arc_span / 2))

        # 词节点 (中间直线)
        word_nodes = sorted([n for n in nodes if n['type'] == 'word'],
                            key=lambda x: x['degree'], reverse=True)
        num_words = len(word_nodes)
        target_height = arc_total_height * 0.9

        for i, node in enumerate(word_nodes):
            t = i / (num_words - 1) if num_words > 1 else 0
            y_val = (t ** stretch_factor) * target_height - (target_height / 2)
            coords[node['id']] = (0, round(y_val, 2))

        # 其他节点 (两侧弧形)
        right_factor = 0.9
        other_nodes = [n for n in nodes if n['type'] != 'word']
        mid = len(other_nodes) // 2
        groups = [other_nodes[:mid], other_nodes[mid:]]
        configs = [{"c_x": 0, "dir": 1}, {"c_x": 0, "dir": -1}]

        for idx, items in enumerate(groups):
            conf = configs[idx]
            n_items = len(items)
            for r_idx, node in enumerate(items):
                t = r_idx / (n_items - 1) if n_items > 1 else 0.5
                t_shift = 2 * t - 1
                t_warp = math.copysign(abs(t_shift) ** (1 / right_factor), t_shift)
                angle = -(t_warp * (arc_span / 2))
                x = conf["c_x"] + conf["dir"] * (arc_radius * math.cos(angle))
                y = arc_radius * math.sin(angle)
                coords[node['id']] = (round(x, 2), round(y, 2))

        return coords

    @staticmethod
    def calculate_spring_layout(G):
        """力导向布局"""
        return nx.spring_layout(G, iterations=100, seed=42)

    @staticmethod
    def calculate_kamada_kawai_layout(G):
        """KK布局 (适合展示路径结构)"""
        return nx.kamada_kawai_layout(G)

    @staticmethod
    def calculate_grid_layout(G):
        """蛇形网格布局 (适合展示超长路径)"""
        nodes = list(G.nodes())
        n = len(nodes)
        if n == 0: return {}

        # 尝试按路径顺序排列
        try:
            endpoints = [x for x in G.nodes() if G.degree(x) == 1]
            start_node = endpoints[0] if endpoints else nodes[0]
            ordered_nodes = list(nx.dfs_preorder_nodes(G, source=start_node))
            # 如果 ordered_nodes 没包含所有点(非连通)，补全
            if len(ordered_nodes) < n:
                remain = list(set(nodes) - set(ordered_nodes))
                ordered_nodes.extend(remain)
        except:
            ordered_nodes = nodes

        side = math.ceil(math.sqrt(n))
        # 适当调宽一点，变长方形
        cols = math.ceil(side * 1.5)

        coords = {}
        spacing = 100

        for i, node in enumerate(ordered_nodes):
            row = i // cols
            col = i % cols

            # 蛇形
            if row % 2 == 1:
                col = cols - 1 - col

            x = (col - cols/2) * spacing
            y = (row - side/2) * spacing * -1 # 翻转Y轴

            coords[node] = (round(x, 2), round(y, 2))

        return coords

    def get_layout(self, G, layout_type='embracing', stretch_factor=None):
        """布局工厂方法"""
        # 构造节点列表供 Embracing 布局使用
        if layout_type == 'embracing':
            degrees = dict(G.degree())
            layout_nodes = [
                {'id': n, 'degree': degrees[n], 'type': 'word' if 'word' in n else 'other'}
                for n in G.nodes()
            ]
            return self.calculate_embracing_layout(layout_nodes, stretch_factor)
        elif layout_type == 'grid':
            return self.calculate_grid_layout(G)
        elif layout_type == 'kamada_kawai':
            return self.calculate_kamada_kawai_layout(G)
        else:
            return self.calculate_spring_layout(G)

class LongestPathCalculator:
    """负责计算图中的最长路径（近似长路径）以及相关统计信息"""

    @staticmethod
    def get_longest_path(G):
        """
        计算图的近似最长简单路径 (Longest Simple Path Heuristic)
        策略: 使用 DFS 树寻找深度最大的路径
        
        返回: (length_in_edges, path_nodes_list)
        """
        if not G.nodes():
            return 0, []

        # 1. 获取最大连通分量
        if nx.is_connected(G):
            G_comp = G
        else:
            components = nx.connected_components(G)
            largest_cc_nodes = max(components, key=len)
            G_comp = G.subgraph(largest_cc_nodes).copy()

        best_path = []

        try:
            # 2. 寻找候选起始点
            # 策略：选取图中距离较远的点（伪外围节点）作为 DFS 起点
            # 如果计算 periphery 太慢，可以改用 heuristic 方法
            if len(G_comp) > 500:
                # 大图使用近似算法找端点
                try:
                    endpoints = nx.approximation.diameter(G_comp) # 返回 (d, u, v) 不对，networkx版本不同返回不同
                    # 兼容性写法：随机取点找最远
                    # 这里简单处理：取度数最小的几个点作为起点候选
                    degrees = sorted(G_comp.degree, key=lambda x: x[1])
                    candidates = [n for n, d in degrees[:5]]
                except:
                    # Fallback
                    candidates = [list(G_comp.nodes())[0]]
            else:
                try:
                    candidates = nx.periphery(G_comp)
                    # 限制数量
                    if len(candidates) > 5:
                        candidates = candidates[:5]
                except:
                    candidates = [list(G_comp.nodes())[0]]

            # 3. 对每个候选点构建 DFS 树并找最长路径
            for start_node in candidates:
                # DFS 树可以将图展开为深层结构，避免短路环
                # nx.dfs_tree 返回的是一个有向无环图 (DAG)
                T = nx.dfs_tree(G_comp, source=start_node)

                # 在 DAG 中找最长路径是线性的
                try:
                    path = nx.dag_longest_path(T)
                    if len(path) > len(best_path):
                        best_path = path
                except Exception:
                    continue

        except Exception as e:
            print(f"Error calculating longest path: {e}")
            return 0, []

        return len(best_path) - 1, best_path

    @staticmethod
    def get_diameter_value(G):
        """计算真实的图直径值"""
        if not G.nodes(): return 0
        if nx.is_connected(G):
            G_comp = G
        else:
            largest_cc_nodes = max(nx.connected_components(G), key=len)
            G_comp = G.subgraph(largest_cc_nodes)

        try:
            # 对于 2000 节点以下的图，直接算 diameter 还是很快的
            if G_comp.number_of_nodes() < 3000:
                return nx.diameter(G_comp)
            else:
                return nx.approximation.diameter(G_comp)
        except:
            return 0

    @staticmethod
    def get_path_stats(G, path_nodes, df_subset):
        """计算路径子图的统计信息"""
        stats = GraphAnalyzer.get_basic_stats(G, df_subset)
        stats['path_node_count'] = len(path_nodes)
        return stats

class LargestCycleCalculator:
    """负责计算图中的最大环（The Longest Cycle）以及相关统计信息"""

    @staticmethod
    def get_longest_cycle(G):
        """
        计算图中的最大简单环 (Heuristic)
        环是在同一个连通分量中寻找的。
        二分图的环长度一定是偶数。
        
        策略: 
        1. 获取基本循环基 (Cycle Basis)。
        2. 既然基本循环基只能找到基础环，对于大图找最长环是NP-Hard。
        3. 采用 Heuristic: 寻找基本环，并尝试组合或在 DFS 树中寻找返祖边构成的长环。
        但为了效率和实现复杂度，这里使用 NetworkX 的 minimum_cycle_basis 先找基，然后找其中最长的（通常是很短的）。
        如果要找真的长环，可以尝试：
        - 从最长路径算法改造，如果是 diameter 的端点 u, v，如果有一条边 (u, v) 则构成环。
        - 或者移除一条边后找最长路径。
        
        这里采用一种实用策略：
        寻找 cycle_basis 中的最长者。注意：这对大图可能不够完美，但对于展示结构通常已足够捕捉局部的大环结构。
        如果需要更激进的 Heuristic，可以对 MST 添加边产生的的基本环进行搜索。
        """
        if not G.nodes():
            return 0, []

        # 1. 获取最大连通分量
        if nx.is_connected(G):
            G_comp = G
        else:
            components = nx.connected_components(G)
            largest_cc_nodes = max(components, key=len)
            G_comp = G.subgraph(largest_cc_nodes).copy()

        try:
            # 使用 cycle_basis (寻找基本环集合)
            # 这是一个多项式时间算法
            cycles = nx.cycle_basis(G_comp)

            if not cycles:
                return 0, []

            # 找到最长的环
            longest_cycle = max(cycles, key=len)
            return len(longest_cycle), longest_cycle

        except Exception as e:
            print(f"Error calculating longest cycle: {e}")
            return 0, []

    @staticmethod
    def get_cycle_stats(G, cycle_nodes, df_subset):
        """计算环子图的统计信息"""
        stats = GraphAnalyzer.get_basic_stats(G, df_subset)
        stats['cycle_node_count'] = len(cycle_nodes)
        return stats

class ComplexNetworkCalculator:
    """负责计算复杂网络特征指标"""

    @staticmethod
    def calculate_metrics(G, df_subset=None):
        """
        计算图的各项网络指标
        """
        if G is None or G.number_of_nodes() == 0:
            return {}

        # 1. 基础指标
        num_nodes = G.number_of_nodes()
        num_edges = G.number_of_edges()
        is_connected = nx.is_connected(G)
        density = nx.density(G)

        # Count node types
        word_nodes_count = len([n for n in G.nodes() if 'word' in str(n)])
        song_nodes_count = len([n for n in G.nodes() if 'song' in str(n)])

        # 2. 度指标
        degrees_dict = dict(G.degree())
        degree_values = sorted(list(degrees_dict.values()))
        max_degree = max(degree_values) if degree_values else 0
        avg_degree = sum(degree_values) / num_nodes if num_nodes > 0 else 0

        # Helper to get name
        def get_node_name(node_id):
            if df_subset is None:
                return str(node_id)
            try:
                if 'word' in str(node_id):
                    # 优化：如果df很大，这样查询可能慢，但只查Top20可以接受
                    # 假定 df_subset 包含 word_id 和 word 列
                    match = df_subset[df_subset['word_id'] == node_id]
                    if not match.empty:
                        return str(match['word'].values[0])
                elif 'song' in str(node_id):
                    # 假定 df_subset 包含 song_id_unique 和 song_name_unique
                    match = df_subset[df_subset['song_id_unique'] == node_id]
                    if not match.empty:
                        return str(match['song_name_unique'].values[0])
            except:
                pass
            return str(node_id)

        # Helper for distribution stats
        def get_dist_stats(node_degree_list):
            if not node_degree_list:
                return {}

            values = [d for n, d in node_degree_list]
            s = pd.Series(values)
            desc = s.describe()

            # Get Min/Max nodes (top 3)
            min_val = desc['min']
            max_val = desc['max']

            min_nodes = [n for n, d in node_degree_list if d == min_val]
            max_nodes = [n for n, d in node_degree_list if d == max_val]

            # Resolve names
            min_names = [get_node_name(n) for n in min_nodes[:3]]
            max_names = [get_node_name(n) for n in max_nodes[:3]]

            # Power law estimation (discrete MLE approximation)
            # alpha = 1 + n / sum(ln(x / (xmin - 0.5)))
            v_arr = np.array([v for v in values if v > 0])
            alpha = 0
            if len(v_arr) > 5:
                xmin = np.min(v_arr)
                if xmin > 0.5:
                    term = np.sum(np.log(v_arr / (xmin - 0.5)))
                    if term > 0:
                        alpha = 1 + len(v_arr) / term

            skew = s.skew()
            kurt = s.kurt()

            dist_type = "Unknown"
            if abs(skew) < 0.5:
                dist_type = "Normal-like"
            elif skew > 1:
                dist_type = "Heavy-tailed"
                if alpha > 1.5 and alpha < 3.5:
                    dist_type = "Power-law-like"
            elif skew > 0:
                dist_type = "Right-skewed"

            # Generate description
            dist_map = {
                "Unknown": "未知",
                "Normal-like": "类正态",
                "Heavy-tailed": "重尾",
                "Power-law-like": "类幂律",
                "Right-skewed": "右偏"
            }
            dist_cn = dist_map.get(dist_type, dist_type)

            desc_text = (
                f"网络度分布呈现{dist_cn}特征，平均度为 {desc['mean']:.2f} (中位数: {desc['50%']})。"
                f"分布偏度为 {skew:.2f}，峰度为 {kurt:.2f}。"
            )
            if alpha > 0:
                desc_text += f"估计的幂律指数(alpha)为 {alpha:.2f}。"

            desc_text += f"最大度为 {int(max_val)}，对应节点：{', '.join(max_names)}"
            if len(max_nodes) > 3:
                desc_text += f" (以及其他 {len(max_nodes)-3} 个)"
            desc_text += "。"

            return {
                "mean": float(desc['mean']),
                "min": float(desc['min']),
                "max": float(desc['max']),
                "median": float(desc['50%']),
                "std": float(desc['std']),
                "kurtosis": float(kurt) if not math.isnan(kurt) else 0.0,
                "skewness": float(skew) if not math.isnan(skew) else 0.0,
                "distribution_type": dist_type,
                "power_law_exponent": float(alpha),
                "min_nodes_names": min_names,
                "max_nodes_names": max_names,
                "description": desc_text
            }

        # Separated lists
        word_nodes_list = [(n, d) for n, d in degrees_dict.items() if 'word' in str(n)]
        song_nodes_list = [(n, d) for n, d in degrees_dict.items() if 'song' in str(n)]

        degree_values_words = sorted([d for n, d in word_nodes_list])
        degree_values_songs = sorted([d for n, d in song_nodes_list])

        max_degree_words = max(degree_values_words) if degree_values_words else 0
        avg_degree_words = np.mean(degree_values_words) if degree_values_words else 0

        max_degree_songs = max(degree_values_songs) if degree_values_songs else 0
        avg_degree_songs = np.mean(degree_values_songs) if degree_values_songs else 0

        # Stats
        # Need pass full list with IDs to helper now
        all_nodes_list = list(degrees_dict.items())
        stats_overall = get_dist_stats(all_nodes_list)
        stats_words = get_dist_stats(word_nodes_list)
        stats_songs = get_dist_stats(song_nodes_list)

        top_10_words_raw = sorted(word_nodes_list, key=lambda x: x[1], reverse=True)[:10]
        top_10_songs_raw = sorted(song_nodes_list, key=lambda x: x[1], reverse=True)[:10]

        top_10_words = [{"id": n, "name": get_node_name(n), "degree": d} for n, d in top_10_words_raw]
        top_10_songs = [{"id": n, "name": get_node_name(n), "degree": d} for n, d in top_10_songs_raw]

        # 3. 连通性指标
        num_connected_components = nx.number_connected_components(G)
        avg_shortest_path_length = -1
        diameter = -1

        # 如果不连通，计算最大连通子图的路径指标
        if num_nodes > 0:
            if is_connected:
                calc_G = G
            else:
                largest_cc_nodes = max(nx.connected_components(G), key=len)
                calc_G = G.subgraph(largest_cc_nodes)

            # 为了性能，大图时可能需要略过
            if calc_G.number_of_nodes() < 3000:
                try:
                    avg_shortest_path_length = nx.average_shortest_path_length(calc_G)
                    diameter = nx.diameter(calc_G)
                except Exception as e:
                    print(f"Metrics calc warning (path len): {e}")

        # 4. 中心性指标 (Centrality)
        # 注意: 结果是字典 {node_id: value}
        try:
            # 主要是度中心性，也就够了
            degree_centrality = nx.degree_centrality(G)
        except:
            degree_centrality = {}

        try:
            # 介数中心性计算较慢，大图使用近似
            if num_nodes > 2000:
                betweenness_centrality = nx.betweenness_centrality(G, k=min(200, num_nodes))
            else:
                betweenness_centrality = nx.betweenness_centrality(G)
        except:
            betweenness_centrality = {}

        # 5. 简单的排序后的Top N展示数据 (便于查看，虽然Raw data也在)
        top_degree = sorted(degrees_dict.items(), key=lambda x: x[1], reverse=True)[:10]

        # Generate basic stats description
        conn_str = "连通" if is_connected else "非连通"
        basic_desc = (
            f"该网络包含 {num_nodes} 个节点（{word_nodes_count} 个词节点，{song_nodes_count} 个歌曲节点）"
            f"和 {num_edges} 条边。网络密度为 {density:.5f}。"
            f"图呈{conn_str}状态，包含 {num_connected_components} 个连通分量。"
        )
        if diameter != -1:
            basic_desc += f"最大连通子图的直径为 {diameter}，平均最短路径长度为 {avg_shortest_path_length:.2f}。"

        return {
            "basic_stats": {
                "node_count": num_nodes,
                "word_nodes_count": word_nodes_count,
                "song_nodes_count": song_nodes_count,
                "edge_count": num_edges,
                "density": density,
                "is_connected": is_connected,
                "connected_components_count": num_connected_components,
                "diameter": diameter,
                "average_shortest_path_length": avg_shortest_path_length,
                "description": basic_desc
            },
            "degree_stats": {
                "max_degree": max_degree,
                "avg_degree": avg_degree,
                "degree_distribution": degree_values,
                "degree_distribution_words": degree_values_words,
                "degree_distribution_songs": degree_values_songs,
                "max_degree_words": max_degree_words,
                "avg_degree_words": float(avg_degree_words),
                "max_degree_songs": max_degree_songs,
                "avg_degree_songs": float(avg_degree_songs),
                "stats_overall": stats_overall,
                "stats_words": stats_words,
                "stats_songs": stats_songs,
                "top_10_degree_nodes": top_degree,
                "top_10_words": top_10_words,
                "top_10_songs": top_10_songs
            },
            "centrality_metrics": {
                "degree_centrality": degree_centrality,
                "betweenness_centrality": betweenness_centrality
            }
        }

class GraphAnalyzer:
    """负责图的统计指标分析"""

    @staticmethod
    def get_diameter_data(G):
        """(Deprecated) 请使用 LongestPathCalculator"""
        return LongestPathCalculator.get_longest_path(G)

    @staticmethod
    def get_basic_stats(G, df_subset):
        """计算基础统计量：节点数、专辑覆盖数等"""
        word_nodes = [n for n in G.nodes() if 'word' in n]
        song_nodes = [n for n in G.nodes() if 'song' in n]

        # 统计 Song 节点所属的专辑数量
        album_ids = set()
        if not df_subset.empty and 'song_id_unique' in df_subset.columns:
            # 筛选出属于当前图节点的歌曲行
            valid_songs = df_subset[df_subset['song_id_unique'].isin(song_nodes)]
            album_ids = set(valid_songs['album_id_unique'])

        return {
            'word_node_count': len(word_nodes),
            'song_node_count': len(song_nodes),
            'covered_album_count': len(album_ids)
        }

# --- 4. 数据构建与导出层 ---


class GraphJsonBuilder:
    """负责将图对象转换为前端可视化 JSON 数据"""

    def __init__(self, data_loader, output_dir):
        self.loader = data_loader
        self.output_dir = output_dir

    def build_json(self,
                   G,
                   df_subset,
                   pos_coords,
                   pos_type,
                   unit_str,
                   raw_total_count,
                   word_num,
                   diameter=None,
                   stats=None,
                   title_suffix=""):
        """构建完整的图数据 JSON 结构"""
        degrees = dict(G.degree())
        word_song_counts = None
        if 'song_id' in df_subset.columns:
            word_song_counts = df_subset.groupby(
                'word_id')['song_id'].count().to_dict()

        graph_data = defaultdict(list)

        # Calculate min/max size for normalization
        max_size = 1
        min_size = 1
        if degrees:
            sizes = list(degrees.values())
            max_size = max(sizes)
            min_size = min(sizes)

        # 1. Nodes
        for node in G.nodes():
            info = defaultdict(str)
            info['id'] = node
            s = degrees.get(node, 0)
            info['size'] = s
            # Normalize to 1-5
            if max_size > min_size:
                info['size_show'] = 1 + 4 * (s - min_size) / (max_size -
                                                              min_size)
            else:
                info['size_show'] = 3  # default middle if all same

            if node in pos_coords:
                info['x'] = pos_coords[node][0]
                info['y'] = pos_coords[node][1]

            self._enrich_node_info(info, node, df_subset, word_song_counts)
            graph_data['nodes'].append(info)

        # 2. Edges
        for u, v in G.edges():
            edge = defaultdict(str)
            edge['source'] = u
            edge['target'] = v
            if 'weight' in G[u][v]:
                edge['weight'] = G[u][v]['weight']

            self._enrich_edge_info(edge, u, v, df_subset)
            graph_data['edges'].append(edge)

        # 3. Metadata
        graph_data['data_info'] = self._generate_metadata(
            df_subset, pos_type, unit_str, raw_total_count, word_num, diameter,
            stats, title_suffix)

        return graph_data

    def _enrich_node_info(self, info, node_id, df, word_song_counts):
        """填充节点详细属性"""
        if 'word' in node_id:
            match = df[df['word_id'] == node_id]
            info['label'] = match['word'].values[
                0] if not match.empty else str(node_id)
            info['node_type'] = 'word'
            info['data'] = {'cluster': '词'}
            if word_song_counts and node_id in word_song_counts:
                info['song_id_count'] = int(word_song_counts[node_id])

        elif 'song' in node_id:
            info['node_type'] = 'song'
            self._fill_meta(info, node_id, df, 'song_id_unique')

        elif 'album' in node_id:
            info['node_type'] = 'album'
            self._fill_meta(info, node_id, df, 'album_id_unique')

    def _fill_meta(self, info, node_id, df, id_col):
        match = df[df[id_col] == node_id]
        if not match.empty:
            row = match.iloc[0]
            info.update({
                'label':
                row['song_name_unique']
                if 'song' in node_id else row['album_fixed'],
                'album_id':
                row['album_id_unique'],
                'album':
                row['album_fixed'],
                'album_order':
                int(row['album_order']),
                'data': {
                    'cluster': row['album_fixed']
                }
            })
            if self.loader.is_ost and 'tv_name' in row:
                info['tv_name'] = row['tv_name']

    def _enrich_edge_info(self, edge, u, v, df):
        """填充边属性 (主要是 Album 信息)"""
        target = u if 'word' not in u else v
        if 'word' in target: return

        col = 'song_id_unique' if 'song' in target else 'album_id_unique'
        match = df[df[col] == target]
        if not match.empty:
            edge['album'] = match.iloc[0]['album_fixed']
            edge['album_order'] = int(match.iloc[0]['album_order'])

    def _generate_metadata(self,
                           df,
                           pos_type,
                           unit_str,
                           raw_total,
                           word_num,
                           diameter=None,
                           stats=None,
                           title_suffix=""):
        """生成元数据"""
        pos_labels = {'n': '名词', 'a': '形容词', 'v': '动词', 't': '时间词'}
        singer = df['artist_name'].values[0] if not df.empty else "Unknown"

        has_pos_count = 0

        if unit_str == "张":
            has_pos_count = df['album_id_unique'].nunique()
        else:
            # if 'song_id_unique' in df.columns:
            has_pos_count = df['song_id_unique'].nunique()
        # elif 'album_id_unique' in df.columns:
        #     has_pos_count = df['album_id_unique'].nunique()

        is_high_freq = "高频" if word_num else ""
        pos_label = pos_labels.get(pos_type, pos_type)
        unit_str_trans = unit_str.replace("首", "歌曲").replace("张", "专辑")
        nodes_type = f'{pos_label}-{unit_str_trans}'
        chart_title = f"{singer}歌词 {is_high_freq}{nodes_type}关系图"
        is_cover_all = f"涵盖全部{unit_str_trans}。" if has_pos_count == raw_total else f"涵盖{has_pos_count}{unit_str}{unit_str_trans}。"
        if word_num:
            chart_sub_title = f"共统计{raw_total}{unit_str}{unit_str_trans}，覆盖量(含该词的歌曲数)最高的{word_num}个{pos_label}，{is_cover_all}"
            chart_sub_title_lcc = ""
        else:
            if stats:
                chart_sub_title = f"共统计{raw_total}{unit_str}{unit_str_trans}，全图包含{stats.get('song_node_count', 0)}个歌曲节点和全部的{stats.get('word_node_count', 0)}个{pos_label}节点，{stats.get('edge_count', 0)}条边"
                chart_sub_title_lcc = f"共统计{raw_total}{unit_str}{unit_str_trans}，最大连通子图包含{stats.get('song_node_count', 0)}个歌曲节点和{stats.get('word_node_count', 0)}个{pos_label}节点，{stats.get('edge_count', 0)}条边"
            else:
                chart_sub_title = ""
                chart_sub_title_lcc = ""
        meta = {
            'singer': singer,
            'title': self.loader.album_type_desc + title_suffix,
            'pos': pos_label,
            'all_num': raw_total,
            'has_pos_num': has_pos_count,
            'word_num': word_num if word_num else "All",
            'num_unit': unit_str,
            'nodes_type': nodes_type,
            'chart_title': chart_title,
            'chart_sub_title': chart_sub_title,
            'chart_sub_title_lcc': chart_sub_title_lcc
        }

        if diameter is not None:
            if "最长路径" in title_suffix:
                meta['path_length'] = diameter
                meta['path_desc'] = "最长路径图"
                if stats:
                    wc = stats.get('word_node_count', 0)
                    sc = stats.get('song_node_count', 0)
                    ac = stats.get('covered_album_count', 0)
                    meta['description'] = (
                        f"该最长路径长度为 {diameter}，连接了 {wc} 个{pos_labels.get(pos_type, pos_type)}节点"
                        f"和 {sc} 首歌曲，跨越了 {ac} 张专辑。路径展示了词汇与歌曲之间的深度关联链条。")
            elif "最大环" in title_suffix:
                meta['cycle_length'] = diameter
                meta['cycle_desc'] = "最大环图"
                if stats:
                    wc = stats.get('word_node_count', 0)
                    sc = stats.get('song_node_count', 0)
                    ac = stats.get('covered_album_count', 0)
                    meta['description'] = (
                        f"该最大环长度为 {diameter}，包含 {wc} 个{pos_labels.get(pos_type, pos_type)}节点"
                        f"和 {sc} 首歌曲，涉及 {ac} 张专辑。这种闭合结构反映了词汇在不同歌曲间的高频共现与其形成的语义闭环。"
                    )
            else:
                meta['diameter'] = diameter
                # 兼容前端显示的直径描述（最大连接组件）
                meta['diameter_desc'] = "最大连通子图直径"

        if stats:
            meta.update({
                'word_node_count':
                stats.get('word_node_count', 0),
                'song_node_count':
                stats.get('song_node_count', 0),
                'covered_album_count':
                stats.get('covered_album_count', 0)
            })
            # Add all_nodes_num
            total_nodes = stats.get('word_node_count', 0) + stats.get(
                'song_node_count', 0)
            # If explicit node_count is available (e.g. from ComplexNetworkCalculator)
            if 'node_count' in stats:
                total_nodes = stats['node_count']
            meta['all_nodes_num'] = total_nodes

            if 'original_node_count' in stats:  # 直径子图特有
                meta['nodes_count_diameter'] = stats['original_node_count']
            if 'path_node_count' in stats:
                meta['path_node_count'] = stats['path_node_count']
            if 'cycle_node_count' in stats:
                meta['cycle_node_count'] = stats['cycle_node_count']

        return meta

    def save_json(self, data, filename):
        path = os.path.join(self.output_dir, filename)
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=4)
        print(f"File generated: {path}")

# --- 5. 管道控制层 ---

class LyricsAnalysisPipeline:
    """管道控制器：串联数据加载、计算、图构建与导出"""

    def __init__(self, data_path, is_ost=False):
        self.loader = LyricsDataLoader(data_path, is_ost)
        self.metrics = MetricsCalculator(self.loader)
        self.layout_engine = LayoutEngine()
        self.exporter = GraphJsonBuilder(self.loader, data_path)
        self.analyzer = GraphAnalyzer() # 新增分析器实例
        self.path_calculator = LongestPathCalculator() # 新增路径计算器
        self.cycle_calculator = LargestCycleCalculator() # 新增最大环计算器
        self.metrics_calculator = ComplexNetworkCalculator() # 新增复杂网络指标计算器

    def _get_filename_suffix(self, words_num):
        """根据 words_num 生成文件后缀"""
        return "_full" if not words_num else ""

    def run_word_song_graph(self, pos_types=['n', 'a', 'v', 't'], words_num=None, layout_type='embracing', stretch_factor=0.8, output_combined=True, is_starts_with=True):
        """生成词-曲二分图（全图和LCC两个版本）"""
        combined_full = {}  # 全图版本
        combined_lcc = {}   # LCC版本
        total_counts = self.loader.get_total_counts()
        suffix = self._get_filename_suffix(words_num)

        for pos in pos_types:
            print(f"Processing Word-Song [{pos}]...")
            df_sub = self.metrics.get_subset(pos, words_num, is_starts_with=is_starts_with)

            # 构建全图
            G_full = nx.from_pandas_edgelist(df_sub, 'word_id', 'song_id_unique', edge_attr=True, create_using=nx.Graph())

            # 布局 (如果 words_num 为 None，强制使用 spring 布局)
            current_layout = layout_type
            if words_num is None:
                current_layout = 'spring'

            # =========================
            # 1. 生成全图版本
            # =========================
            if G_full.number_of_nodes() > 0:
                print(f"  Full Graph - Nodes: {G_full.number_of_nodes()}, Edges: {G_full.number_of_edges()}")
                coords_full = self.layout_engine.get_layout(G_full, current_layout, stretch_factor)
                true_diameter_full = self.path_calculator.get_diameter_value(G_full)
                stats_full = self.analyzer.get_basic_stats(G_full, df_sub)
                stats_full['edge_count'] = G_full.number_of_edges()

                json_data_full = self.exporter.build_json(
                    G_full, df_sub, coords_full, pos, '首', total_counts['song_count'], words_num,
                    diameter=true_diameter_full, stats=stats_full
                )

                if not output_combined:
                    self.exporter.save_json(json_data_full, f'{pos}_word_song_graph_data_full{suffix}.json')
                combined_full[pos] = json_data_full

            # =========================
            # 2. 生成LCC版本
            # =========================
            G_lcc = G_full.copy()
            if G_lcc.number_of_nodes() > 0:
                # 提取最大连通分量 (LCC)
                largest_cc_nodes = max(nx.connected_components(G_lcc), key=len)
                G_lcc = G_lcc.subgraph(largest_cc_nodes).copy()
                print(f"  LCC Graph - Nodes: {G_lcc.number_of_nodes()}, Edges: {G_lcc.number_of_edges()}")

                coords_lcc = self.layout_engine.get_layout(G_lcc, current_layout, stretch_factor)
                true_diameter_lcc = self.path_calculator.get_diameter_value(G_lcc)
                stats_lcc = self.analyzer.get_basic_stats(G_lcc, df_sub)
                stats_lcc['edge_count'] = G_lcc.number_of_edges()

                json_data_lcc = self.exporter.build_json(
                    G_lcc, df_sub, coords_lcc, pos, '首', total_counts['song_count'], words_num,
                    diameter=true_diameter_lcc, stats=stats_lcc
                )

                if not output_combined:
                    self.exporter.save_json(json_data_lcc, f'{pos}_word_song_graph_data_full_lcc{suffix}.json')
                combined_lcc[pos] = json_data_lcc

        if output_combined:
            # 分别输出全图和LCC版本
            # 注：suffix 已包含 _full，所以直接使用
            full_suffix = suffix if suffix else ""
            self.exporter.save_json(combined_full, f'graph_word_song_data{full_suffix}.json')
            self.exporter.save_json(combined_lcc, f'graph_word_song_data{full_suffix}_lcc.json')

    def run_longest_path_graph(self, pos_types=['n', 'a', 'v', 't'], words_num=None, output_combined=True, is_starts_with=True):
        """生成最长路径（直径）子图数据 (独立输出)"""
        combined = {}
        total_counts = self.loader.get_total_counts()
        suffix = self._get_filename_suffix(words_num)

        for pos in pos_types:
            print(f"Processing Longest Path Graph [{pos}]...")
            df_sub = self.metrics.get_subset(pos, words_num, is_starts_with=is_starts_with)

            # 构建图 (保持一致)
            G = nx.from_pandas_edgelist(df_sub, 'word_id', 'song_id_unique', edge_attr=True, create_using=nx.Graph())

            # 计算最长路径
            diameter, diameter_path_nodes = self.path_calculator.get_longest_path(G)

            if diameter > 0 and diameter_path_nodes:
                # 仅构建路径本身的边，而不是导出子图（导出子图会包含路径节点间的所有边）
                G_path = nx.Graph()
                # 添加路径边
                path_edges = list(zip(diameter_path_nodes[:-1], diameter_path_nodes[1:]))
                for u, v in path_edges:
                    # 尝试保留原图边的属性
                    edge_attr = G[u][v] if G.has_edge(u, v) else {}
                    G_path.add_edge(u, v, **edge_attr)

                # 确保节点也被添加（虽然add_edge会自动添加，但为了保险）
                G_path.add_nodes_from(diameter_path_nodes)

                # 路径子图布局：使用 grid 布局来展示路径结构
                path_coords = self.layout_engine.get_layout(G_path, 'grid')

                # 获取详细统计信息
                path_stats = self.path_calculator.get_path_stats(G_path, diameter_path_nodes, df_sub)

                path_json_data = self.exporter.build_json(
                    G_path, df_sub, path_coords, pos, '首',
                    total_counts['song_count'], word_num="Longest Path",
                    diameter=diameter, stats=path_stats, title_suffix=" (最长路径)"
                )

                if not output_combined:
                    self.exporter.save_json(path_json_data, f'{pos}_longest_path_graph_data{suffix}.json')
                combined[pos] = path_json_data
            else:
                combined[pos] = {}

        if output_combined:
            self.exporter.save_json(combined, f'graph_longest_path_data{suffix}.json')

    def run_largest_cycle_graph(self, pos_types=['n', 'a', 'v', 't'], words_num=None, output_combined=True, is_starts_with=True):
        """生成最大环（Cycle）子图数据 (独立输出)"""
        combined = {}
        total_counts = self.loader.get_total_counts()
        suffix = self._get_filename_suffix(words_num)

        for pos in pos_types:
            print(f"Processing Largest Cycle Graph [{pos}]...")
            df_sub = self.metrics.get_subset(pos, words_num, is_starts_with=is_starts_with)

            # 构建图
            G = nx.from_pandas_edgelist(df_sub, 'word_id', 'song_id_unique', edge_attr=True, create_using=nx.Graph())

            # 计算最大环
            cycle_len, cycle_nodes = self.cycle_calculator.get_longest_cycle(G)

            if cycle_len > 0 and cycle_nodes:
                # 仅构建环本身的边，而不是导出子图
                G_cycle = nx.Graph()

                # 添加环的边 (闭合路径)
                # cycle_nodes 是有序的节点列表
                edges = list(zip(cycle_nodes[:-1], cycle_nodes[1:])) + [(cycle_nodes[-1], cycle_nodes[0])]

                for u, v in edges:
                    # 尝试保留原图边的属性
                    edge_attr = G[u][v] if G.has_edge(u, v) else {}
                    G_cycle.add_edge(u, v, **edge_attr)

                G_cycle.add_nodes_from(cycle_nodes)

                # 环布局：使用 circular 或 kamada_kawai 布局能较好展示环状
                cycle_coords = nx.circular_layout(G_cycle)
                # 或者使用 kamada_kawai_layout 获得更自然的展开
                # cycle_coords = nx.kamada_kawai_layout(G_cycle)

                # 获取详细统计信息
                cycle_stats = self.cycle_calculator.get_cycle_stats(G_cycle, cycle_nodes, df_sub)

                cycle_json_data = self.exporter.build_json(
                    G_cycle, df_sub, cycle_coords, pos, '首',
                    total_counts['song_count'], word_num="Largest Cycle",
                    diameter=cycle_len, stats=cycle_stats, title_suffix=" (最大环)"
                )

                if not output_combined:
                    self.exporter.save_json(cycle_json_data, f'{pos}_largest_cycle_graph_data{suffix}.json')
                combined[pos] = cycle_json_data
            else:
                combined[pos] = {}

        if output_combined:
            self.exporter.save_json(combined, f'graph_largest_cycle_data{suffix}.json')

    def run_network_metrics_analysis(self, pos_types=['n', 'a', 'v', 't'], words_num=None, output_combined=True, is_starts_with=True):
        """生成复杂网络指标数据"""
        combined = {}
        suffix = self._get_filename_suffix(words_num)

        for pos in pos_types:
            print(f"Calculating Network Metrics [{pos}]...")
            df_sub = self.metrics.get_subset(pos, words_num, is_starts_with=is_starts_with)

            # 构建图
            G = nx.from_pandas_edgelist(df_sub, 'word_id', 'song_id_unique', edge_attr=True, create_using=nx.Graph())

            # 计算指标
            metrics = self.metrics_calculator.calculate_metrics(G, df_sub)

            # 补充一些 Metadata
            metrics['meta'] = self.exporter._generate_metadata(
                df_sub, pos, '指标', self.loader.get_total_counts()['song_count'],
                words_num, list(G.degree()) if G else [], # 这里的参数传递有些hacky, 不过 _generate_metadata 不太依赖这个列表
                title_suffix=" (网络指标)"
            )

            if not output_combined:
                self.exporter.save_json(metrics, f'{pos}_network_metrics{suffix}.json')
            combined[pos] = metrics

        if output_combined:
            self.exporter.save_json(combined, f'network_metrics_data{suffix}.json')

    def run_word_album_graph(self, pos_types=['n', 'a', 'v', 't'], words_num=None, layout_type='spring', output_combined=True, is_starts_with=True):
        """生成词-专辑二分图"""
        combined = {}
        total_counts = self.loader.get_total_counts()
        suffix = self._get_filename_suffix(words_num)

        for pos in pos_types:
            print(f"Processing Word-Album [{pos}]...")
            df_sub = self.metrics.get_subset(pos, words_num, is_starts_with=is_starts_with)

            # 聚合权重
            df_weighted = df_sub.groupby(['word_id', 'album_id_unique']).size().reset_index(name='weight')
            G = nx.from_pandas_edgelist(df_weighted, 'word_id', 'album_id_unique', edge_attr='weight', create_using=nx.Graph())

            # 布局 (通常 Spring)
            coords = self.layout_engine.get_layout(G, layout_type, stretch_factor=None)

            # 构建 JSON
            json_data = self.exporter.build_json(
                G, df_sub, coords, pos, '张', total_counts['album_count'], words_num
            )

            if not output_combined:
                self.exporter.save_json(json_data, f'{pos}_word_album_graph_data{suffix}.json')
            combined[pos] = json_data

        if output_combined:
            self.exporter.save_json(combined, f'graph_word_album_data{suffix}.json')

    def run_matrix_analysis(self, pos_types=['n', 'v', 'a', 't'], words_num=None, is_starts_with=True):
        """生成词频矩阵"""
        print("Generating Matrix...")
        result = {}

        for pos in pos_types:
            df_sub = self.metrics.get_subset(pos, words_num, is_starts_with=is_starts_with)
            if df_sub.empty:
                result[pos] = {'matrixData': [], 'albumNames': [], 'wordNames': []}
                continue

            # 透视 & 排序
            df_matrix = df_sub.groupby(['album_name', 'word']).size().unstack(fill_value=0)

            album_dates = df_sub.groupby('album_name')['publish_date'].first()
            valid_albums = [a for a in album_dates.sort_values().index if a in df_matrix.index]
            df_matrix = df_matrix.loc[valid_albums]
            df_matrix = df_matrix[df_matrix.sum().sort_values(ascending=False).index]

            matrix_list = [
                [col, idx, int(df_matrix.loc[idx, col])]
                for col in df_matrix.columns
                for idx in df_matrix.index
            ]

            result[pos] = {
                'matrixData': matrix_list,
                'albumNames': list(df_matrix.index),
                'wordNames': list(df_matrix.columns),
                'chartTitle': f'高频{LyricProcessConstants.POS_DICT[pos]}-专辑内歌曲覆盖数量矩阵'
            }

        self.exporter.save_json(result, 'matrix_album_word.json')

    def run_cloud_words_analysis(self, pos_types=['n', 'a', 'v', 't'], words_num=20, is_starts_with=True):
        """生成词云数据 (高频词统计)"""
        print("Generating Cloud Words Data...")
        df_raw = self.loader.load()

        # 获取歌手名称
        singer_name = df_raw['artist_name'].iloc[0] if not df_raw.empty else "Unknown"

        # 获取 POS_DICT (词性字典)
        pos_dict = getattr(LyricProcessConstants, 'POS_DICT', {
            'n': '名词', 'a': '形容词', 'v': '动词', 't': '时间词'
        })

        words_dict = defaultdict(dict)

        for pos in pos_types:
            # 统计指定词性的高频词 (自动过滤停用词)
            res_df = self.metrics._count_word_stats(df_raw, pos, words_num, is_starts_with, filter_stopwords=True)

            if res_df.empty:
                words_dict[pos] = {
                    'word': [],
                    'song_num': [],
                    'freq': [],
                    'order': [],
                    'chart_title': f"{singer_name}-覆盖量(含该词的歌曲数)最高的{words_num}个{pos_dict.get(pos, pos)}"
                }
            else:
                words_dict[pos] = res_df.to_dict('list')
                words_dict[pos]['chart_title'] = f"{singer_name}-覆盖量(含该词的歌曲数)最高的{words_num}个{pos_dict.get(pos, pos)}"

        # 保存为 JSON
        self.exporter.save_json(dict(words_dict), 'cloud_words_data.json')

# --- Main 执行入口 ---

def main(file_path_prefix="", words_num_=None, is_starts_with_=False, pos_types_=['n', 'a', 'v', 't'], cloud_words_num_=20):
    """
    main 的 Docstring
    
    :param file_path_prefix: 数据文件路径前缀
    :param words_num_: 要处理的词的数量，None 表示处理所有词
    :param is_starts_with_: 是否只处理以指定词性开头的词
    :param pos_types_: 要处理的词性列表，例如 ['n', 'a', 'v', 't']
    :param cloud_words_num_: 词云数据中要显示的高频词数量
    """


    script_dir = os.path.dirname(os.path.abspath(__file__))
    full_path_prefix = os.path.join(script_dir, file_path_prefix)

    # 实例化管道
    # is_ost=True 会按年份聚合专辑 ID
    pipeline = LyricsAnalysisPipeline(full_path_prefix, is_ost=False)

    # 执行任务

    # 1. 词-歌曲图
    # words_num=0 或 None 表示提取所有词
    # layout_type 可选 'embracing' 或 'spring'
    pipeline.run_word_song_graph(
        pos_types=pos_types_,
        words_num=words_num_,
        layout_type='embracing',
        output_combined=True,
        is_starts_with=is_starts_with_
    )

    # 2. 最长路径 (直径) 子图 (独立)
    pipeline.run_longest_path_graph(
        pos_types=pos_types_,
        words_num=words_num_,
        output_combined=True,
        is_starts_with=is_starts_with_
    )

    # 3. 最大环 子图 (独立)
    pipeline.run_largest_cycle_graph(
        pos_types=pos_types_,
        words_num=words_num_,
        output_combined=True,
        is_starts_with=is_starts_with_
    )

    # 4. 复杂网络指标
    pipeline.run_network_metrics_analysis(
        pos_types=pos_types_,
        words_num=words_num_,
        output_combined=True,
        is_starts_with=is_starts_with_
    )

    # 5. 词-专辑图
    pipeline.run_word_album_graph(
        pos_types=pos_types_,
        words_num=words_num_,
        layout_type='spring',
        output_combined=True,
        is_starts_with=is_starts_with_
    )

    # 6. 矩阵
    pipeline.run_matrix_analysis(
        pos_types=pos_types_,
        words_num=20,
        is_starts_with=is_starts_with_
    )

    # 7. 词云数据（高频词统计）
    pipeline.run_cloud_words_analysis(
        pos_types=pos_types_,
        words_num=cloud_words_num_,
        is_starts_with=is_starts_with_
    )

if __name__ == "__main__":
    # 配置区
    # file_path_prefix = "data/jaychou/"
    file_path_prefix = "data/mayday/"
    # file_path_prefix = "data/liuyuning/"
    words_num = 20
    is_starts_with = False
    pos_types = ['n', 'a', 'v', 't']
    cloud_words_num = 20  # 词云数据中要显示的高频词数量

    main(
        file_path_prefix=file_path_prefix,
        words_num_=words_num,
        is_starts_with_=is_starts_with,
        pos_types_=pos_types,
        cloud_words_num_=cloud_words_num
    )
