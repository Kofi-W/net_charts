import streamlit as st
import pandas as pd

# Page config
st.set_page_config(page_title="周杰伦歌曲指标分析", layout="wide")

# Title
st.title("🎵 周杰伦歌曲极值分析")
st.markdown("---")

# Data
metrics_data = {
    '最长的歌': '以父之名',
    '最长的歌时长': '5分42秒',
    'longest_song_number': 342,
    '最短的歌': '超跑女神',
    '最短的歌时长': '2分35秒',
    'shortest_song_number': 155,
    '前奏最长': '半兽人',
    '前奏最长时长': '63秒',
    'longest_intro_number': 63,
    '前奏最短': '公主病',
    '前奏最短时长': '0秒',
    'shortest_intro_number': 0,
    '歌词字数最多': '以父之名',
    '歌词字数最多字数': '1226字',
    'most_lyrics_number': 1226,
    '歌词字数最少': '蒲公英的约定',
    '歌词字数最少字数': '227字',
    'least_lyrics_number': 227,
    '歌词重复率最高': '可爱女人',
    '歌词重复率最高重复率': '82.31%',
    '歌词重复率最低': '最伟大的作品',
    '歌词重复率最低重复率': '38.67%',
    '歌名最长': 'Now You See Me',
    '歌名最长字数': '11字',
    'longest_song_name_number': 11
}

# Song Duration Section
st.subheader("📊 歌曲时长分析")
col1, col2 = st.columns(2)

with col1:
    st.metric(
        label="⏱️ 最长的歌",
        value=metrics_data['最长的歌'],
        delta=metrics_data['最长的歌时长']
    )

with col2:
    st.metric(
        label="⏱️ 最短的歌",
        value=metrics_data['最短的歌'],
        delta=metrics_data['最短的歌时长']
    )

# Intro Analysis
st.subheader("🎼 前奏时长分析")
col3, col4 = st.columns(2)

with col3:
    st.metric(
        label="🔊 前奏最长",
        value=metrics_data['前奏最长'],
        delta=metrics_data['前奏最长时长']
    )

with col4:
    st.metric(
        label="🔊 前奏最短",
        value=metrics_data['前奏最短'],
        delta=metrics_data['前奏最短时长']
    )

# Lyrics Analysis
st.subheader("📝 歌词字数分析")
col5, col6 = st.columns(2)

with col5:
    st.metric(
        label="✍️ 歌词字数最多",
        value=metrics_data['歌词字数最多'],
        delta=metrics_data['歌词字数最多字数']
    )

with col6:
    st.metric(
        label="✍️ 歌词字数最少",
        value=metrics_data['歌词字数最少'],
        delta=metrics_data['歌词字数最少字数']
    )

# Repetition Rate Analysis
st.subheader("🔄 歌词重复率分析")
col7, col8 = st.columns(2)

with col7:
    st.metric(
        label="📈 重复率最高",
        value=metrics_data['歌词重复率最高'],
        delta=metrics_data['歌词重复率最高重复率']
    )

with col8:
    st.metric(
        label="📉 重复率最低",
        value=metrics_data['歌词重复率最低'],
        delta=metrics_data['歌词重复率最低重复率']
    )

# Song Name Length
st.subheader("🎤 歌名长度分析")
col9, _ = st.columns(2)

with col9:
    st.metric(
        label="📌 歌名最长",
        value=metrics_data['歌名最长'],
        delta=metrics_data['歌名最长字数']
    )

st.markdown("---")
st.caption("数据来源：周杰伦歌曲极值统计")