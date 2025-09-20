from collections import defaultdict
from operator import itemgetter
import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import math

# 计算两组经纬度点之间距离
from math import radians, cos, sin, asin, sqrt

import sys

sys.path.append('..')
from k_libs.db_query import DBOperate

# 数据指标
# 1. 换乘线路最多的站点，线路数、直接连通站点数
# 2. 站点最多的线路, 站点数, 线路名称
# 3. 里程最长的线路, 路线里程, 线路名称
# 4. 经过站点最多的路线，不重复站点数
# 5. 距离最长的路线，路线里程
# 5. 经过站点最多的环线，不重复站点数
# 6. 里程最长的环线，路线里程
# 7. 距离最长的两个相邻站点，站点名称、线路名称、距离

# 高德地图经纬度转百度地图经纬度
def gcj02_to_bd09(lng, lat):
    x_pi = 3.14159265358979324 * 3000.0 / 180.0
    x = lng
    y = lat
    z = math.sqrt(x ** 2 + y ** 2) + 0.00002 * math.sin(y * x_pi)
    theta = math.atan2(y, x) + 0.000003 * math.cos(x * x_pi)
    bd_lng = z * math.cos(theta) + 0.0065
    bd_lat = z * math.sin(theta) + 0.006
    return bd_lng, bd_lat


# 球面距离计算
class GeoDistance:
    def __init__(self):
        pass

    # LBS 球面距离公式
    def geodistance(self, lng1, lat1, lng2, lat2):
        lng1, lat1, lng2, lat2 = map(
            radians,
            [float(lng1), float(lat1),
             float(lng2), float(lat2)])  # 经纬度转换成弧度
        dlon = lng2 - lng1
        dlat = lat2 - lat1
        a = sin(dlat / 2)**2 + cos(lat1) * cos(lat2) * sin(dlon / 2)**2
        distance = 2 * asin(sqrt(a)) * 6371 * 1000  # 地球平均半径，6371km
        distance = round(distance / 1000, 3)
        return distance

    # LBS 球面距离公式 用于df
    def geodistance_df(self, df, lng1, lat1, lng2, lat2):
        df = df.fillna(0)
        lng1, lat1, lng2, lat2 = map(radians, [
            float(df[lng1]),
            float(df[lat1]),
            float(df[lng2]),
            float(df[lat2])
        ])  # 经纬度转换成弧度
        if not 0 in [lng1, lat1, lng2, lat2]:
            dlon = lng2 - lng1
            dlat = lat2 - lat1
            a = sin(dlat / 2)**2 + cos(lat1) * cos(lat2) * sin(dlon / 2)**2
            distance = 2 * asin(sqrt(a)) * 6371 * 1000  # 地球平均半径，6371km
            distance = round(distance / 1000, 3)
            return distance


# 从数据库查询地铁数据，临时用于测试
class NetDataQuery:
    def __init__(self, db_info):
        self.db_info = db_info

    def query_data(self):
        sql = "SELECT * FROM ads_subway;"
        df = DBOperate(self.db_info).read_sql(sql)
        return df

    def st_pos(self):
        df_st = self.query_datay()
        df_st = df_st[[
            'city_name', 'st_name', 'st_id', 'pos_x', 'pos_y', 'longitude',
            'latitude'
        ]].drop_duplicates()
        return df_st

    def city_line(self):
        sql = "SELECT * FROM ads_subway_line;"
        df = DBOperate(self.db_info).read_sql(sql)
        return df


# 含支线的线路数据处理
class BranchDataClear:
    def __init__(self, df):
        self.df = df

    # 统计存在支线的线路，即线路名称相同，但线路ID不同
    def branch_line_count(self):
        branch_line_cnt = pd.pivot_table(
            self.df,
            values='line_id',
            index=['city_name', 'line_name'],
            aggfunc=pd.Series.nunique).reset_index()
        branch_line_cnt = branch_line_cnt.rename(
            columns={'line_id': 'line_id_cnt'})
        branch_line_cnt = branch_line_cnt[
            branch_line_cnt['line_id_cnt'] > 1].reset_index(drop=True)
        return branch_line_cnt

    # 数据处理
    def branch_data_clear(self):
        df = self.df.copy()
        # 原始数据中，当某一线路存在支线时，该线路中的共用站点会重复出现，需要删除这种重复数据
        df = df.sort_values(by=['city_id', 'line_id'])
        df = df.drop_duplicates(
            subset=['city_id', 'line_name', 'st_id', 'target_st_id'],
            keep='first')
        # 有支线线路的line_number会有多个，为统计城市线路时对线路排序，需要统一这类线路的line_number，使用同一线路最小的line_number值，作为uni_line_number
        line_df = df[['city_id', 'line_number', 'line_name']]
        line_df = line_df.drop_duplicates(subset=['city_id', 'line_name'],
                                          keep='first')
        # 为原始数据补充uni_line_number列
        df_new = df.merge(line_df, how='left', on=['city_id', 'line_name'])
        df_new = df_new.rename(columns={'line_number_y': 'uni_line_number'})
        return df_new


# 数据清洗
class NetDataClear:
    def __init__(self, df_db):
        # df_db:原始数据，ads_subway
        self.df_db = df_db

    # 添加站间距离
    @staticmethod
    def add_distance(df):
        # 基于两个站点经纬度计算距离
        df['ll_distance'] = df.apply(GeoDistance().geodistance_df,
                                  axis=1,
                                  lng1='longitude',
                                  lat1='latitude',
                                  lng2='target_st_longitude',
                                  lat2='target_st_latitude')
        return df

    # 支线数据处理
    @staticmethod
    def branch_clear(df):
        return BranchDataClear(df).branch_data_clear()
    
    # 城市名称处理
    @staticmethod
    def city_name_clear(df):
        df = df.copy()
        df['city_name'] = df['city_name'].apply(lambda x: x.replace('市', ''))
        df['city_name'] = df['city_name'].apply(lambda x: x.replace('特别行政区', ''))
        return df
    
    # 过滤掉已弃用的路线
    @staticmethod
    def filter_abandoned_line(df):
        df = df.copy()
        df = df[df['line_is_show'] == '1']
        return df
    
    # 经纬度转换，高德转百度
    @staticmethod
    def lnglat_to_bd09(df):
        df = df.copy()
        df['longitude'] = df['longitude'].astype(float)
        df['latitude'] = df['latitude'].astype(float)
        df['longitude_bd'], df['latitude_bd'] = zip(*df.apply(lambda x: gcj02_to_bd09(x['longitude'], x['latitude']), axis=1))
        df['target_st_longitude'] = df['target_st_longitude'].astype(float)
        df['target_st_latitude'] = df['target_st_latitude'].astype(float)
        df['target_st_longitude_bd'], df['target_st_latitude_bd'] = zip(*df.apply(lambda x: gcj02_to_bd09(x['target_st_longitude'], x['target_st_latitude']), axis=1))
        return df
    
    # 删除起始站和终点站相同的数据
    @staticmethod
    def remove_same_st(df):
        df = df.copy()
        df = df[df['st_name'] != df['target_st_name']]
        return df

    # 特殊城市数据处理
    @staticmethod
    def special_city_clear(df):
        # 并线数据处理
        # 删除成都19号线，st_name为三岔、福田、天府机场1号2号航站楼、天府机场北的数据
        df = df.copy()
        df = df[~((df['city_name'] == '成都') & (df['line_name'] == '19号线') & (df['st_name'].isin(['三岔', '福田', '天府机场1号2号航站楼', '天府机场北'])))]
        return df
    
    # 线路排序，添加line_order列
    @staticmethod
    def line_order(df):
        df = df.copy()
        # 新增line_order列，如果line_name不以数字开头，则使用uni_line_number+100为line_order，否则使用uni_line_number
        df['line_order'] = df.apply(lambda x: x['uni_line_number'] + 100 if not str(x['line_name']).startswith(tuple('0123456789')) else x['uni_line_number'], axis=1)
        return df

    # 数据清洗主函数
    def data_clear(self):
        df = self.df_db.copy()
        df = self.filter_abandoned_line(df)
        df = self.branch_clear(df)
        df = self.city_name_clear(df)
        df = self.lnglat_to_bd09(df)
        df = self.remove_same_st(df)
        df = self.add_distance(df)
        df = self.special_city_clear(df)
        df = self.line_order(df)
        return df


# 数据透视
class NetDataPivot:
    def __init__(self, df, distance_col='ll_distance'):
        # df: 可以是整体数据，也可以是单个城市数据
        # distance_col: 里程字段，默认使用ll_distance，即基于经纬度计算的距离；如果是使用官方的距离数据，则传入'official_distance'
        self.df = df.copy()
        self.df['distance'] = self.df[distance_col]

    # 整体数据统计透视
    def overall_pivot(self):
        # df: 整体数据
        overall_cnt = pd.pivot_table(
            self.df,
            values=['line_name', 'st_id', 'distance'],
            index=['city_name'],
            aggfunc={
                'line_name': 'nunique',
                'st_id': 'nunique',
                'distance': 'sum'
            },
        )
        overall_cnt = overall_cnt.reset_index().rename(columns={
            'line_name': 'line_cnt',
            'st_id': 'st_cnt'
        })
        overall_cnt = overall_cnt.sort_values(
            by='line_cnt', ascending=False).reset_index(drop=True)
        return overall_cnt

    def city_data(self, city_name):
        # df: 整体数据
        df = self.df[self.df['city_name'] == city_name]
        return df

    # 统计单个城市线路总里程与站点数
    def city_pivot(self):
        # df: 单个城市数据
        df = self.df.copy()
        city_cnt = pd.pivot_table(df,
                                  values=['st_id', 'distance'],
                                  index=['uni_line_number', 'line_name'],
                                  aggfunc={
                                      'st_id': 'nunique',
                                      'distance': 'sum'
                                  })
        city_cnt = city_cnt.reset_index().rename(columns={'st_id': 'line_st_cnt', 'distance': 'line_distance'})
        return city_cnt[['line_name', 'line_st_cnt', 'line_distance']]

    # 统计单个城市站点所在线路数
    def city_st_line_cnt(self):
        # df: 单个城市数据
        df = self.df.copy()
        st_line_cnt = pd.pivot_table(df,
                                     values='line_id',
                                     index=['st_id'],
                                     aggfunc='nunique').reset_index()
        st_line_cnt = st_line_cnt.rename(columns={'line_id': 'conn_line_cnt'})
        return st_line_cnt


# 生成图数据,df: 单个城市数据
class NetDataGraph:
    def __init__(self, df, distance_col='ll_distance'):
        # df: 单个城市数据
        # distance_col: 距离字段
        self.df = df.copy()
        self.df['distance'] = self.df[distance_col]

    # 普通图数据对象
    def generate_G(self):
        # 剔除target_st_name为空的行
        df_g = self.df[self.df['target_st_poiid'].notnull()]
        # 删除st_name == target_st_name的行
        df_g = df_g[df_g['st_poiid'] != df_g['target_st_poiid']]
        G = nx.from_pandas_edgelist(df_g,
                                    source='st_poiid',
                                    target='target_st_poiid',
                                    edge_attr=['line_color', 'distance', 'line_id'],
                                    create_using=nx.Graph())
        return G
    
    # 无向多重图数据对象
    def generate_MultiG(self):
        # 剔除target_st_name为空的行
        df_g = self.df[self.df['target_st_poiid'].notnull()]
        # 删除st_name == target_st_name的行
        df_g = df_g[df_g['st_poiid'] != df_g['target_st_poiid']]
        G = nx.from_pandas_edgelist(df_g,
                                    source='st_poiid',
                                    target='target_st_poiid',
                                    edge_attr=['line_color', 'distance', 'line_id'],
                                    create_using=nx.MultiGraph())
        return G

    # 站点距离字典
    def st_distance_dict(self):
        df = self.df.copy()
        st_distance = df[['st_poiid', 'target_st_poiid',
                          'distance']].drop_duplicates()
        st_distance_dict = defaultdict()
        for row in st_distance.iterrows():
            st_distance_dict[(row[1]['st_poiid'],
                              row[1]['target_st_poiid'])] = row[1]['distance']
            st_distance_dict[(row[1]['target_st_poiid'],
                              row[1]['st_poiid'])] = row[1]['distance']
        return st_distance_dict


# 度相关指标
class NetDegree:
    def __init__(self, G):
        self.G = G

    # 节点的度，字典形式
    def nodes_degree(self):
        return dict(self.G.degree())

    # 平均度
    def avg_degree(self):
        return np.mean(list(dict(self.G.degree()).values()))

    # 密度
    def density(self):
        return nx.density(self.G)

    # 同配系数
    def assortativity(self):
        return nx.degree_assortativity_coefficient(self.G)

    # 集聚系数
    def clustering(self):
        return nx.average_clustering(self.G)

    # 度中心性
    def degree_centrality(self):
        return nx.degree_centrality(self.G)

    # 度分布
    def degree_distribution(self):
        degree_sequence = sorted([d for n, d in self.G.degree()], reverse=True)
        degreeCount = defaultdict(int)
        for degree in degree_sequence:
            degreeCount[degree] += 1
        return degreeCount

    # 度分布图
    def degree_distribution_plot(self):
        degree_sequence = sorted([d for n, d in self.G.degree()], reverse=True)
        degreeCount = defaultdict(int)
        for degree in degree_sequence:
            degreeCount[degree] += 1
        deg, cnt = zip(*degreeCount.items())
        fig, ax = plt.subplots()
        plt.bar(deg, cnt, width=0.80, color='b')
        plt.title("Degree Histogram")
        plt.ylabel("Count")
        plt.xlabel("Degree")
        ax.set_xticks([d + 0.4 for d in deg])
        ax.set_xticklabels(deg)
        plt.show()

    # 中心性指标
    def centrality_dict(self):
        res = {'degree_centrality': self.degree_centrality()}
        return res


# 直径、效率和平均最短距离
# 直径：网络中任意两点间距离的最大值，即地铁网络中，从任意一个站点出发，都可以经过不超过(直径-1)个站点，到达任意一个站点。
class NetLength:
    def __init__(self, G, weight=None):
        self.G = G
        self.weight = weight

    # 直径
    def diameter(self):
        return nx.diameter(self.G)

    # 加权直径
    def diameter_weighted(self):
        return nx.diameter(self.G, weight=self.weight)

    # 平均路径长度
    def avg_path_lenght(self):
        return nx.average_shortest_path_length(self.G)

    # 加权平均路径长度
    def avg_path_lenght_weighted(self):
        return nx.average_shortest_path_length(self.G, weight=self.weight)

    # 局部效率
    def local_efficiency(self):
        return nx.local_efficiency(self.G)

    # 全局效率
    def global_efficiency(self):
        return nx.global_efficiency(self.G)

    # 指标字典
    def length_dict(self):
        if self.weight:
            res = {
                'diameter': self.diameter(),
                'diameter_weighted': self.diameter_weighted(),
                'avg_path_lenght': self.avg_path_lenght(),
                'avg_path_lenght_weighted': self.avg_path_lenght_weighted(),
                'local_efficiency': self.local_efficiency(),
                'global_efficiency': self.global_efficiency()
            }
        else:
            res = {
                'diameter': self.diameter(),
                'avg_path_lenght': self.avg_path_lenght(),
                'local_efficiency': self.local_efficiency(),
                'global_efficiency': self.global_efficiency()
            }
        return res


# 全局最短路径，定位直径经过的节点路径
class NetPath:
    paths = None
    paths_unique = None
    paths_unique_s = None
    paths_unique_s_weighted = None
    path_d_list = None
    path_d_list_weighted = None

    def __init__(self, G, weight=None, is_weighted=False):
        self.G = G
        self.weight = weight
        self.is_weighted = is_weighted
        self.all_shortest_paths = self.get_paths()
        self.clear_paths_data()

    def get_paths(self):
        if self.is_weighted:
            return dict(nx.shortest_path(self.G, weight=self.weight))
        else:
            return dict(nx.shortest_path(self.G))

    def clear_paths_data(self):
        paths = []
        paths_unique = []
        path_d_list = []
        path_d_list_weighted = []
        # 使用字典验证节点对是否已存在，提高效率
        paths_d = defaultdict(int)
        # 加权最短路径距离
        p = dict(nx.shortest_path_length(
            self.G, weight=self.weight)) if self.weight else None
        for s, t in self.all_shortest_paths.items():
            for s1, t1 in t.items():
                # 节点对，排序
                node_pair = sorted([s, s1])
                # 距离 =节点数-1
                path_distance = len(t1) - 1
                path_distance_weighted = p[s][s1] if self.weight else 0
                # 路径列表，([起点, 终点], 距离)
                path = (node_pair, path_distance, path_distance_weighted)
                paths.append(path)
                self.paths = paths
                # 去重，起点 != 终点
                if node_pair[0] != node_pair[1]:
                    # 节点对文本拼接作为字典的key
                    key = node_pair[0] + node_pair[1]
                    # 如果字典的key不存在，说明尚未添加进列表
                    if not paths_d[key] == 1:
                        # 去重路径列表，([起点, 终点], 距离)
                        paths_unique.append(path)
                        self.paths_unique = paths_unique
                        # 节点距离列表
                        path_d_list.append(path_distance)
                        path_d_list_weighted.append(path_distance_weighted)
                        self.path_d_list = path_d_list
                        self.path_d_list_weighted = path_d_list_weighted
                        # 字典key值设为1
                        paths_d[key] = 1
        # 按距离降序排列
        self.paths_unique_s = sorted(paths_unique,
                                     key=itemgetter(1),
                                     reverse=True)
        # 按距离降序排列_加权
        self.paths_unique_s_weighted = sorted(paths_unique,
                                              key=itemgetter(2),
                                              reverse=True)

    # 所有最短路径的距离的直方图
    def get_path_d_hist(self, by_weight=False):
        d_list = self.path_d_list
        if by_weight:
            d_list = self.path_d_list_weighted
        high = max(d_list)
        bins = 50 if high > 50 else int(high)
        fig, ax = plt.subplots()
        ax.hist(d_list, bins=bins, rwidth=0.9)
        ax.set_ylabel('Count')
        ax.set_xlabel('Distance')
        if by_weight:
            ax.set_xlabel('Weighted Distance')
        ax.set_title("Histogram of Distances for All Shortest Paths ")
        return fig

    # 两点距离=直径的节点
    def get_longest_paths(self, by_weight=False):
        longest_paths = []
        if by_weight:
            l = max(self.path_d_list_weighted)
            for i in self.paths_unique_s_weighted:
                if i[2] == l:
                    longest_paths.append(i[0])
        else:
            l = max(self.path_d_list)
            for i in self.paths_unique_s:
                if i[1] == l:
                    longest_paths.append(i[0])
        return longest_paths

    # 两点距离=直径的节点的路径
    def get_longest_paths_edges(self, by_weight=False):
        max_path_edges = []
        longest_paths = self.get_longest_paths(by_weight=by_weight)
        for i in longest_paths:
            max_path = self.all_shortest_paths[i[0]][i[1]]
            max_path_edges.append([(v, max_path[i + 1])
                                   for i, v in enumerate(max_path)
                                   if i < len(max_path) - 1])
        return max_path_edges

    # 两点最短路径
    def get_path_by_node(self, source, target):
        path = self.all_shortest_paths[source][target]
        path_edges = [(v, path[i + 1]) for i, v in enumerate(path)
                      if i < len(path) - 1]
        return path_edges


# 环
class NetCycle:
    # 按站点数全局最大环线
    max_chain_by_st = None
    # 站点数
    max_chain_by_st_cnt = None
    # 环线里程
    max_chain_by_st_distance = None
    # 按线路距离全局最大环线
    max_chain_by_distance = None
    # 站点数
    max_chain_by_distance_st_cnt = None
    # 环线里程
    max_chain_by_distance_distance = None

    def __init__(self, G, df, distance_col='ll_distance'):
        # df: 单个城市数据
        self.G = G
        self.df = df.copy()
        self.df['distance'] = self.df[distance_col]
        self.global_max_chain()
    
    def st_distance_dict(self):
        # 站点间距离字典
        df = self.df.copy()
        st_distance = df[['st_id', 'target_st_id', 'distance']].drop_duplicates()
        st_distance_dict = defaultdict()
        for row in st_distance.iterrows():
            st_distance_dict[(row[1]['st_id'], row[1]['target_st_id'])] = row[1]['distance']
            st_distance_dict[(row[1]['target_st_id'], row[1]['st_id'])] = row[1]['distance']
        return st_distance_dict
    
    def global_max_chain(self):
        st_distance_dict = self.st_distance_dict()
        # 全局最大环线
        chains = list(nx.chain_decomposition(self.G))
        # 按站点数
        self.max_chain_by_st = max(chains, key=len)
        self.max_chain_by_st_cnt = len(self.max_chain_by_st)
        self.max_chain_by_st_distance = sum([st_distance_dict[self.max_chain_by_st[i]] for i in range(len(self.max_chain_by_st))])
        # 按线路距离
        chain_distance_l = []
        for chain in chains:
            chain_distance = sum([
                st_distance_dict[chain[i]] for i in range(len(chain))
            ])
            chain_distance_l.append(chain_distance)
        print(chain_distance_l)
        self.max_chain_by_distance_distance = max(chain_distance_l)
        max_index = chain_distance_l.index(self.max_chain_by_distance_distance)
        self.max_chain_by_distance = chains[max_index]
        self.max_chain_by_distance_st_cnt = len(self.max_chain_by_distance)


    # 全局最大环线,按站点数
    def global_max_chain_by_st_cnt(self):
        chains = list(nx.chain_decomposition(self.G))
        print(chains)
        max_chain = max(chains, key=len)
        return max_chain

    # 指定节点最大环线，如果该节点不在环线内，则返回距离该节点最近节点的环线
    def node_max_chain(self, root_node):
        node_chains = list(nx.chain_decomposition(self.G, root=root_node))
        node_max_chain = max(node_chains, key=len)
        return node_max_chain

    # 全局最大环线,按线路距离
    def global_max_chain_by_distance(self):
        # 按里程最大环线
        chains = list(nx.chain_decomposition(self.G))
        chain_distance_l = []
        for chain in chains:
            chain_distance = sum([
                self.st_distance_dict[chain[i]] for i in range(len(chain) - 1)
            ])
            chain_distance_l.append(chain_distance)
        max_index = chain_distance_l.index(max(chain_distance_l))
        max_chain = chains[max_index]
        return max_chain
    

def gcj02_to_bd09(lng, lat):
    x_pi = 3.14159265358979324 * 3000.0 / 180.0
    x = lng
    y = lat
    z = math.sqrt(x ** 2 + y ** 2) + 0.00002 * math.sin(y * x_pi)
    theta = math.atan2(y, x) + 0.000003 * math.cos(x * x_pi)
    bd_lng = z * math.cos(theta) + 0.0065
    bd_lat = z * math.sin(theta) + 0.006
    return bd_lng, bd_lat


# 计算质心的函数
def calculate_centroid(coords):
    # coords 是一个包含 x, y 坐标的元组列表
    x_coords = [coord[0] for coord in coords]
    y_coords = [coord[1] for coord in coords]
    center_x = np.mean(x_coords)
    center_y = np.mean(y_coords)
    return (center_x, center_y)

# 使用 move_towards_centroid 函数将每个子组的质心移动到一个更接近整体质心的位置，具体通过 factor 参数控制移动距离。 factor=0.5 意味着将质心移动一半距离，使其接近但不覆盖整体质心。
# 计算并应用平移向量，使用线性插值
def move_towards_centroid(group, overall_centroid, factor):
    # group 是一个包含 x, y 坐标的元组列表
    # overall_centroid 是整体质心的坐标
    # factor 是一个介于 0 和 1 之间的值，控制移动距离。 factor=0.5 意味着将质心移动一半距离
    group_centroid = calculate_centroid(group)
    translation_vector = ((overall_centroid[0] - group_centroid[0]) * factor,
                          (overall_centroid[1] - group_centroid[1]) * factor)
    moved_group = [(x + translation_vector[0], y + translation_vector[1]) for (x, y) in group]
    return moved_group


class CityNetworkAnalyzer:
    def __init__(self, df):
        self.df = df

    def analyze(self):
        df_line_st = NetDataPivot(self.df).overall_pivot()
        city_list = df_line_st['city_name'].unique().tolist()
        city_net_index = []
        for city in city_list:
            city_index_dict = {}
            city_index_dict['city_name'] = city
            df_city = self.df[self.df['city_name'] == city]
            G_city = NetDataGraph(df_city).generate_G()
            city_index_dict['连边数'] = G_city.number_of_edges()
            net_d = NetDegree(G_city)
            city_index_dict['平均度'] = float(net_d.avg_degree().round(2))
            city_index_dict['最大度'] = max(dict(G_city.degree()).values())
            city_index_dict['换乘站数量'] = len([i for i in dict(G_city.degree()).values() if i > 2])
            city_index_dict['换乘站比例'] = round(len([i for i in dict(G_city.degree()).values() if i > 2]) / len(G_city.nodes()), 4)
            city_index_dict['密度'] = round(net_d.density(), 5)
            city_index_dict['同配系数'] = round(net_d.assortativity(), 5)
            city_index_dict['平均聚类系数'] = round(net_d.clustering(), 5)
            if nx.is_connected(G_city):
                city_index_dict['是否连通图'] = "是"
            else:
                city_index_dict['是否连通图'] = "否"
                G_city_max_sub = max(nx.connected_components(G_city), key=len)
                G_city = G_city.subgraph(G_city_max_sub)
            net_l = NetLength(G_city)
            city_index_dict['直径'] = net_l.diameter()
            city_index_dict['平均最短路径长度'] = round(net_l.avg_path_lenght(), 2)
            city_index_dict['全局效率'] = round(net_l.global_efficiency(), 5)
            city_index_dict['平均局部效率'] = round(net_l.local_efficiency(), 5)
            city_net_index.append(city_index_dict)

        city_net_index_df = pd.DataFrame(city_net_index)
        df_index = pd.merge(df_line_st, city_net_index_df, on='city_name', how='left')
        df_index = df_index.drop(columns=['distance'])
        df_index = df_index.rename(columns={'city_name': '城市', 'line_cnt': '线路数', 'st_cnt': '车站数'})
        df_index['序号'] = df_index.index + 1
        df_index = df_index[['序号', '城市', '线路数', '车站数', '连边数', '平均度', '最大度', '换乘站数量', '换乘站比例', '密度', '同配系数', '平均聚类系数', '是否连通图', '直径', '平均最短路径长度', '全局效率', '平均局部效率']]
        return df_index


if __name__ == "__main__":
    print("This is a library file and should be imported, not run directly.")
