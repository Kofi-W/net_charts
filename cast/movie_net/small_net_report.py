# streamlit 报告页面
import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(layout="wide")

# 原始数据读取
df = pd.read_csv("youyin_cast_works.csv")
df['k_cast_id'] = df['k_cast_id'].apply(lambda x: str(x))


st.title("电影-影人 网络分析报告")

# 作品概览
st.subheader("作品概览")
movie_list = df[[
    'k_title', 'movie_id_m', 'k_movie_year', 'k_type', 'rating_num',
    'k_rating_people'
]].drop_duplicates().sort_values(by='k_movie_year').reset_index(drop=True)

# 数据预览
table_height = 35 + movie_list.shape[0] * 35
st.data_editor(movie_list, hide_index=True, row_height=35, height=table_height)
# 数据描述
st.markdown(f"""
- 作品总数：{movie_list.shape[0]} 部
- 评分均值：{movie_list['rating_num'].mean():.2f} 分
- 评分中位数：{movie_list['rating_num'].median():.2f} 分
- 评分人数均值：{movie_list['k_rating_people'].mean():.0f} 人
- 评分人数中位数：{movie_list['k_rating_people'].median():.0f} 人
""")
# 评分分布散点图
st.subheader("评分分布散点图")
fig = px.scatter(movie_list,
                 x="k_rating_people",
                 y="rating_num",
                #  size="rating_num",
                 color="k_type",
                 hover_name="k_title",
                 log_x=True,
                 size_max=60)
st.plotly_chart(fig, use_container_width=True)