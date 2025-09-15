import streamlit as st

import pandas as pd

# 读取数据
df_row = pd.read_csv('../data/subway/ads_subway_cleaned.csv')
df_index = pd.read_csv('../data/subway/ads_subway_index.csv')
# 换乘站比例设置为百分比格式
df_index['换乘站比例'] = (df_index['换乘站比例'] * 100).round(2).astype(str) + '%'


st.set_page_config(
    # Title and icon for the browser's tab bar:
    page_title="Seattle Weather",
    page_icon="🌦️",
    # Make the content take up the width of the page:
    layout="wide",
)

# 选择城市
city_list = df_index['城市'].tolist()
choose_city = st.selectbox("选择城市", 
                           city_list, 
                           placeholder="选择城市查看地铁网络结构性指标"
                           )
df_city = df_row[df_row['city_name'] == choose_city]
df_index_city = df_index[df_index['城市'] == choose_city]

st.subheader(f"{choose_city}地铁网络结构性指标")
st.dataframe(df_city)
st.dataframe(df_index_city)

df_index_city_s = df_index_city.iloc[0, 2:]
# 删除值为0的字段
df_index_city_s = df_index_city_s[df_index_city_s != 0]
st.dataframe(df_index_city_s)
# df_index_city_s字段数
index_cols = df_index_city_s.index.shape[0]

with st.container(horizontal=True, gap="medium"):
    

    # index_cols中每2个为一组
    cols_grouped = [(st.container(), st.container()) for _ in range((index_cols + 1) // 2)]

    for group_idx, (col1, col2) in enumerate(cols_grouped):
        cols = st.columns(index_cols, gap="medium")
        i = group_idx * 2
        with col1:
            if i < index_cols:
                col1.metric(label=df_index_city_s.index[i], value=df_index_city_s.iloc[i])
        with col2:
            if i + 1 < index_cols:
                col2.metric(label=df_index_city_s.index[i + 1], value=df_index_city_s.iloc[i + 1])