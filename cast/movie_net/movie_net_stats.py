import streamlit as st

import pandas as pd

st.set_page_config(layout="wide")
# 读取excel文件
# df = pd.read_excel(io='movie_net_stats.xlsx', sheet_name=None)

stats_parts_dict = {
    'df_stats': '原始数据统计',
    'graph_metrics': '整体网络图指标',
    'largest_connected_component': '最大连通子图指标',
    'movie_bipartite_graph_metrics': '作品二分图指标',
    'movie_bipartite_largest_connected_component': '作品二分图最大连通子图指标',
    'cast_bipartite_graph_metrics': '影人二分图指标',
    'cast_bipartite_largest_connected_component': '影人二分图最大连通子图指标'
}

for part_key, part_name in stats_parts_dict.items():
    st.subheader(part_name)
    df_part = df = pd.read_excel(io='movie_network_stats.xlsx',
                                 sheet_name=part_name)
    # df_cols_num = df_part.shape[1]
    table_height = 35 + df_part.shape[0] * 35
    # st.dataframe(df_part)
    st.data_editor(df_part, hide_index=True, row_height=35, height=table_height)

st.subheader("高分电影列表")
df_movie_top_n = pd.read_csv('movie_top_200.csv')
st.data_editor(df_movie_top_n, hide_index=True, row_height=35, height=735)

st.subheader("低分电影列表")
df_movie_top_n_asc = pd.read_csv('movie_lowest_200.csv')
st.data_editor(df_movie_top_n_asc, hide_index=True, row_height=35, height=735)

