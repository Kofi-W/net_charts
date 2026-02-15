# -*- encoding: utf-8 -*-
'''
@File    : songs_libs.py
@Time    : 2026/02/12 17:09:18
@Author  : Kofi Wang
@Contact : wonkefei@gmail.com
'''


def id_process(df, is_ost=False, is_words=True):
    # 曲目，专辑添加唯一id
    df = df.copy()
    df['song_id_unique'] = 'song' + df['song_id'].astype(str)
    # df_album_fixed = df[['album_id', 'album_fixed']].drop_duplicates(subset=['album_fixed'], keep='first').reset_index(drop=True)
    # df_album_fixed['album_id_unique'] = 'album' + df_album_fixed['album_id'].astype(int).astype(str)
    # df_album_fixed = df_album_fixed[['album_id_unique', 'album_fixed']]
    df['song_year'] = df['publish_date'].str.split('-').str[0]
    if is_ost:
        df['album_fixed'] = df['song_year'] + '年'
        df['album_id_unique'] = 'album' + df['song_year']
        df['legend_type'] = df['song_year'] + '年'
    else:
        df['album_fixed'] = df['album_name']
        df['album_id_unique'] = 'album' + df['album_id'].astype(str)
        df['legend_type'] = df['album_name']
    # 生成年份排序映射（album_order）
    # 使用 factorize 可以直接一步生成按顺序排列的编码
    # 如果必须按年份数值排序，则先排序再 factorize
    unique_years = sorted(df['song_year'].unique())
    year_to_order = {year: i for i, year in enumerate(unique_years)}
    df['album_order'] = df['song_year'].map(year_to_order)
    # df['legend_order'] = df['legend_type']
    # 词唯一id
    if is_words:
        df_words = df[['word',
                       'pos']].drop_duplicates().reset_index(drop=False)
        df_words['word_id'] = 'word' + df_words['index'].astype(str)
        df_words = df_words.drop('index', axis=1).reset_index(drop=True)
        df = df.merge(df_words, on=['word', 'pos'], how='left')
    return df


# 常量
class LyricProcessConstants:
    STOP_WORDS = [
        "是", "要", "有", "来", "到", "会", "能", "让", "像", "在", "无", "上", "么", "没",
        "当", "出", "变", "起", "给", "过", "回", "成", "为", "如", "开", "知", "管", "住",
        "讲", "算", "吹", "看", "走", "去", "说", "找", "坐", "停", "懂", "吃", "叫", "落",
        "下", '放'
    ]


if __name__ == "__main__":
    print(
        "This is a library of functions for processing song data, not meant to be run directly."
    )
