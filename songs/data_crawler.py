# -*- encoding: utf-8 -*-
'''
@File    : data_crawler.py
@Time    : 2026/02/24 12:08:58
@Author  : Kofi Wang
@Contact : wonkefei@gmail.com
'''

import requests
import re
import json
import os
import time
import pandas as pd
import html
from datetime import datetime
from collections import defaultdict


from collections import Counter

import sys

sys.path.append('..')


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


def save_to_json_list(file_path, song_data):
    """以列表形式保存所有歌曲，避免字典 key 覆盖的问题"""
    data_list = []
    if os.path.exists(file_path):
        with open(file_path, 'r', encoding='utf-8') as f:
            try:
                data_list = json.load(f)
                if not isinstance(data_list, list): data_list = []
            except:
                data_list = []

    data_list.append(song_data)

    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data_list, f, ensure_ascii=False, indent=4)


# 按关键词请求数据，每页50条
def search_song(keyword, page=0):
    """搜索歌曲并返回歌曲ID"""
    url = "https://c.y.qq.com/soso/fcgi-bin/client_search_cp"
    params = {
        "w": keyword,
        "format": "json",
        "n": 50,
        "p": page,
    }
    headers = {
        "User-Agent":
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }
    response = requests.get(url, params=params, headers=headers)
    if response.status_code == 200:
        data = json.loads(response.text)
        songs = data["data"]["song"]["list"]
        res = []
        for song in songs:
            res_d = {
                "song_id": song["songid"],
                "song_mid": song["songmid"],
                "song_name": song["songname"],
                "song_subname": song["lyric"],
                "artist_name": ",".join([singer["name"] for singer in song["singer"]]),
                "artist_id": ",".join([str(singer["id"]) for singer in song["singer"]]),
                "artist_mid": ",".join([singer["mid"] for singer in song["singer"]]),
                "album_name": song['albumname'],
                "album_id": song['albumid'],
                "album_mid": song['albummid'],
                "duration": song['interval'],
                "publish_time": song["pubtime"],
            }
            res.append(res_d)
        return res
    return []


def get_songs_data_raw(singger, max_page=1):
    """
    获取歌手歌曲列表
    singer_name: 歌手名称
    max_page: 最大页数, 默认每页50条数据，max_page=曲目总数/50
    """
    qq_songs_list = []
    for page in range(0, max_page):
        print(f'正在获取第{page+1}页数据...')
        res = search_song(singger, page)
        time.sleep(2)
        qq_songs_list.extend(res)
    return qq_songs_list


# 数据筛选
# 1. artist_name中包含singger
# 2. song_subname中包含书名号，将书名号中的内容保存为新字段: tv_name
def filter_ost_songs(singger, song_list):
    """
    筛选符合条件的歌曲：
    1. 歌手包含 singger
    2. 子标题包含书名号，并提取书名号内容为 tv_name
    """
    filtered_list = []
    # 预编译正则，匹配《 和 》之间最少的内容
    tv_pattern = re.compile(r'《(.*?)》')

    for song in song_list:
        # 条件 1: 校验 artist_name (确保该字段已在之前的解析中生成)
        if singger not in song.get("artist_name", ""):
            continue

        # 条件 2: 校验 song_name，如果”《“在 song_name 中，则跳过
        if "《" in song.get("song_name", ""):
            continue

        # 条件 2.5: 如果 song_subname 中包含 试听，伴奏，则跳过
        if any(keyword in song.get("song_name", "")
               for keyword in ["试听", "伴奏"]):
            continue

        # 条件 3: 校验 song_subname 并在满足时提取 tv_name
        subname = song.get("song_subname", "")
        match = tv_pattern.search(str(subname))

        if match:
            # 满足条件，创建新字段并保存
            song["tv_name"] = match.group(1)
            filtered_list.append(song)

    return filtered_list

# 歌词采集
def get_qq_lyric(song_id):
    """根据歌曲ID获取歌词"""
    url = "https://c.y.qq.com/lyric/fcgi-bin/fcg_query_lyric_yqq.fcg"
    params = {"nobase64": 1, "musicid": song_id, "format": "json"}
    headers = {
        "Referer":
        "https://y.qq.com/",
        "User-Agent":
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }
    response = requests.get(url, params=params, headers=headers)
    if response.status_code == 200:
        lyric_data = response.json()
        return lyric_data.get("lyric", "")
    return "歌词获取失败"


def get_all_songs_lyric(file_path, songs_df):
    for i, row in songs_df.iterrows():
        song_id = row['song_id']
        song_name = row['song_name']
        # 判断是否已采集
        if os.path.exists(file_path):
            df = pd.read_json(file_path)
            songs_had = df['song_id'].tolist()
            if song_id not in songs_had:
                print(song_name)
                lyric_raw = get_qq_lyric(song_id)
                single_res = {
                    'song_id': song_id,
                    'song_name': song_name,
                    'lyric_raw': lyric_raw
                }
                save_to_json_list(file_path, single_res)
                time.sleep(2)
        else:
            print(song_name)
            lyric_raw = get_qq_lyric(song_id)
            single_res = {
                'song_id': str(song_id),
                'song_name': song_name,
                'lyric_raw': lyric_raw
            }
            save_to_json_list(file_path, single_res)
            time.sleep(2)

# 歌词清洗
class QQLyricCleanerV16:

    def __init__(self):
        self.time_tag_pattern = re.compile(
            r'\[(\d{2,}:?\d{2,}\.\d{2,})\]\s*(.*)')

    def _generate_singer_blacklist(self, singers):
        if not singers: return set()
        # 兼容中英文逗号、斜杠、空格拆分
        names = re.split(r'[,，/ ]', singers)
        blacklist = {'男', '女', '合', '人', '们'}
        for name in names:
            name = name.strip()
            if not name: continue
            blacklist.add(name.upper())
            if len(name) > 1:
                blacklist.add(name[0].upper())  # 姓
        return blacklist

    def _preprocess(self, text):
        if not text: return ""
        text = html.unescape(text)
        return text.replace('\r', '').replace('\\n', '\n')

    def get_credits(self, raw_text):
        text = self._preprocess(raw_text)
        res = {"lyricist": "", "composer": "", "arranger": ""}
        mapping = {
            "lyricist": r"(?:词|作词)\s*[:：]\s*([^\n\r\]]+)",
            "composer": r"(?:曲|作曲)\s*[:：]\s*([^\n\r\]]+)",
            "arranger": r"(?:编曲|制作人|Arranger|Program)\s*[:：]\s*([^\n\r\]]+)"
        }
        for key, pat in mapping.items():
            match = re.search(pat, text, re.IGNORECASE)
            if match:
                res[key] = re.split(r'[\[\]]',
                                    match.group(1).strip())[0].strip()
        return res

    def process_data(self, raw_lyric, song_id=None, song_name="", singers=""):
        singer_bits = self._generate_singer_blacklist(singers)
        credits = self.get_credits(raw_lyric)
        text = self._preprocess(raw_lyric)

        valid_lines = []
        lines = text.split('\n')
        content_line_idx = 0

        for line in lines:
            match = self.time_tag_pattern.search(line)
            if not match: continue

            ts, content = match.groups()
            content = content.strip()
            if not content: continue

            # --- 核心改进：标题/标题行判定 ---
            clean_content = content.replace(" ", "").upper()

            # 识别标题的特征：包含歌曲名、包含歌手名、或者包含大量分隔标点
            contains_singer = any(s in clean_content for s in singer_bits
                                  if len(s) > 1)
            is_metadata_structure = any(
                symbol in content for symbol in ['-', '《', '，', '(', '（'])

            # 判定：如果是前几行，且长得像标题或元数据，直接过滤
            if content_line_idx < 3:
                if (song_name and song_name.upper() in clean_content
                    ) or contains_singer or is_metadata_structure:
                    content_line_idx += 1
                    continue

            # 权利声明过滤
            if any(w in clean_content
                   for w in ['权利保留', '未经许可', '著作权', '版权', '声明', '提供']):
                content_line_idx += 1
                continue

            # --- 冒号处理逻辑 ---
            if ':' in content or '：' in content:
                sep = ':' if ':' in content else '：'
                prefix, *suffix = content.split(sep, 1)
                prefix_clean = prefix.strip().upper()
                suffix_content = "".join(suffix).strip()

                if prefix_clean in singer_bits:
                    if suffix_content:
                        content = suffix_content
                    else:
                        content_line_idx += 1
                        continue
                else:
                    # 只要不在演唱者白名单里的冒号行，全部删除
                    content_line_idx += 1
                    continue

            # 最终清洗
            c_final = re.sub(r'\s+', '，', content).strip('，')
            if c_final:
                valid_lines.append((ts, c_final))
                content_line_idx += 1

        # 数据组装
        start_time = valid_lines[0][0] if valid_lines else ""
        lyrics_text = "。".join([x[1] for x in valid_lines])
        if lyrics_text: lyrics_text += "。"

        return {
            "song_id": song_id,
            "song_name": song_name,
            "start_time": start_time,
            "has_lyric": 1 if lyrics_text else 0,
            "lyricist": credits["lyricist"],
            "composer": credits["composer"],
            "arranger": credits["arranger"],
            "lyrics_text": lyrics_text
        }

# 歌词保存
def clear_and_save_lyric(path_prefix, songs_df):
    lyric_raw = pd.read_json(path_prefix + 'raw_lyric_data.json')

    cleaner = QQLyricCleanerV16()
    res_list = []
    for i, row in songs_df.iterrows():
        song_id = row['song_id']
        song_name = row['song_name']
        singers = row.get('artist_name', "")
        lyric_raw_single = lyric_raw[lyric_raw['song_id'] ==
                                     song_id]['lyric_raw'].values[0]
        single_res = cleaner.process_data(lyric_raw_single, song_id, song_name,
                                          singers)
        res_list.append(single_res)
    with open(path_prefix + 'cleared_lyric_data.json', 'w',
              encoding='utf-8') as f:
        json.dump(res_list, f, ensure_ascii=False, indent=4)



if __name__ == '__main__':
    print("歌词采集器已就位！")