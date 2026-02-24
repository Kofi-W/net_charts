# -*- encoding: utf-8 -*-
'''
@File    : songs_libs.py
@Time    : 2026/02/12 17:09:18
@Author  : Kofi Wang
@Contact : wonkefei@gmail.com
'''
from datetime import datetime


def id_process(df, is_ost=False, is_words=True):
    # 曲目，专辑添加唯一id
    df = df.copy()
    df['song_id_unique'] = 'song' + df['song_id'].astype(str)
    # df_album_fixed = df[['album_id', 'album_fixed']].drop_duplicates(subset=['album_fixed'], keep='first').reset_index(drop=True)
    # df_album_fixed['album_id_unique'] = 'album' + df_album_fixed['album_id'].astype(int).astype(str)
    # df_album_fixed = df_album_fixed[['album_id_unique', 'album_fixed']]
    df['song_year'] = df['publish_date'].str.split('-').str[0].astype(int)
    if is_ost:
        df['album_fixed'] = df['song_year'].astype(str) + '年'
        df['album_id_unique'] = 'album' + df['song_year'].astype(str)
        df['legend_type'] = df['song_year'].astype(str) + '年'
    else:
        df['album_fixed'] = df['album_name']
        df['album_id_unique'] = 'album' + df['album_id'].astype(str)
        df['legend_type'] = df['album_name']
    # 生成年份排序映射（album_order）
    # 使用 factorize 可以直接一步生成按顺序排列的编码
    # 判断是否已存在df['album_order']列
    if 'album_order' not in df.columns:
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

def format_timestamp(ts, date_format='%Y-%m-%d'):
    """
    自动识别秒或毫秒，并转换为指定格式的字符串
    :param ts: 时间戳 (int 或 float)
    """
    if not ts or ts <= 0:
        return "Unknown"

    # 核心逻辑：判断时间戳位数
    # 秒级时间戳目前在 10^9 数量级（10位）
    # 毫秒级时间戳在 10^12 数量级（13位）
    # 我们以 10^11 (11位) 为界限进行区分
    if ts > 100000000000:
        ts = ts / 1000  # 是毫秒，转换为秒

    try:
        dt = datetime.fromtimestamp(ts)
        return dt.strftime(date_format)
    except Exception:
        return "Invalid Date"


class SongDataCleaner:
    """歌曲数据清洗工具类 - 所有方法可独立使用"""

    @staticmethod
    def clear_song_name(df):
        """
        清洗歌曲名称：去空格、统一括号、提取纯名称
        
        :param df: 输入DataFrame（需包含 song_name 列）
        :return: 处理后的DataFrame（新增 song_name_unique 和 song_name_pure 列）
        """
        df = df.copy()
        # 歌曲数据
        df['song_name_unique'] = df['song_name'].astype(str)
        # 清洗song_name_unique，删除空格，将中文括号转为英文括号
        df['song_name_unique'] = df['song_name_unique'].str.replace(' ', '')
        df['song_name_unique'] = df['song_name_unique'].str.replace(
            '（', '(').str.replace('）', ')')
        # 删除括号内的内容
        df['song_name_pure'] = df['song_name_unique'].str.replace(r'\(.*?\)',
                                                                  '',
                                                                  regex=True)
        # 删除空格
        df['song_name_pure'] = df['song_name_pure'].str.replace(' ', '')
        return df

    @staticmethod
    def clear_song_singer(df, singer_name):
        """
        筛选指定歌手的独唱作品
        
        :param df: 输入DataFrame（需包含 artist_name 列）
        :param singer_name: 歌手名称
        :return: 筛选后的DataFrame
        """
        df = df.copy()
        df = df[df['artist_name'] == singer_name]
        return df

    @staticmethod
    def clear_song_tv_show(df, albums):
        """
        删除疑似翻唱/综艺作品（按专辑名前缀筛选）
        
        :param df: 输入DataFrame（需包含 album_name 列）
        :param albums: 要排除的专辑名称列表（支持前缀匹配）
        :return: 筛选后的DataFrame
        """
        df = df.copy()
        df['album_name_pure'] = df['album_name'].astype(str).fillna('无')
        for album in albums:
            df = df[~df['album_name_pure'].str.startswith(album)]
        return df

    @staticmethod
    def generate_vatual_album(df):
        """
        生成虚拟专辑：每10首歌为一个PART
        
        :param df: 输入DataFrame（需包含 album_name 列）
        :return: 处理后的DataFrame（新增 album_name_raw 列，修改 album_name 和 album_order）
        """
        df = df.copy()
        df['album_name_raw'] = df['album_name']
        df = df.drop(columns=['album_name'])
        df['album_name'] = "PART " + (df.index // 10 + 1).astype(str)
        df['album_order'] = df.index // 10
        return df


# 常量
class LyricProcessConstants:

    STOP_WORDS_N = []
    STOP_WORDS_V = [
    '是', '在', '有', '了', '着', '被', '把', '让', '使', '得',
    '要', '会', '能', '可以', '应该', '必须', '不要', '不能', '像', '到', '没有', '就是', '不是', '没', '无', '给', '不会'
    ]
    STOP_WORDS_A = [
    '很', '太', '最', '真', '非常', '十分'
    ]
    STOP_WORDS_T = ['晚安']
    POS_DICT = {
        'n': '名词',
        'a': '形容词',
        'v': '动词',
        't': '时间词'
    }


if __name__ == "__main__":
    print(
        "This is a library of functions for processing song data, not meant to be run directly."
    )
