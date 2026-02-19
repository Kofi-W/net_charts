# 最终的完整代码
import json
from openai import OpenAI

class LyricAnalyzer:
    def __init__(self, api_key: str):
        self.client = OpenAI(
            api_key=api_key,
            base_url="https://api.deepseek.com"
        )
    
    def analyze(self, song_data: dict) -> dict:
        """分析单首歌词"""
        
        prompt = f"""分析歌词，返回分词、词性、词频。

歌词：
{song_data['lyrics_text']}

要求：
- 合理分词，保持语义完整
- 包含词性：n名词、v动词、adj形容词、time时间词等
- 统计每个词的出现频率
- 按频率从高到低排序

返回JSON格式：
{{"res": [
    {{"text": "词", "pos": "n", "pos_desc": "名词", "freq": 次数}},
    ...
]}}"""

        response = self.client.chat.completions.create(
            model="deepseek-chat",
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            temperature=0.1
        )
        
        result = json.loads(response.choices[0].message.content)
        
        return {
            "song_id": song_data["song_id"],
            "res": result.get("res", [])
        }
    
    def batch_analyze(self, songs: list) -> list:
        """批量分析多首歌词"""
        results = []
        for song in songs:
            print(f"分析: {song.get('song_name', song['song_id'])}")
            results.append(self.analyze(song))
        return results

# 使用
if __name__ == "__main__":
    analyzer = LyricAnalyzer(api_key="sk-0d71ac942f3a4b45bc5d2b58efade80a")
    
    song_data = {
        "song_id": 1056501,
        "song_name": "星空",
        "start_time": "00:14.63",
        "has_lyric": 1,
        "lyricist": "阿信",
        "composer": "石头",
        "arranger": "五月天",
        "lyrics_text": "摸不到的颜色，是否叫彩虹。看不到的拥抱，是否叫做微风。一个人，想着一个人。是否就叫寂寞。命运偷走如果，只留下结果。时间偷走初衷，只留下了苦衷。你来过，然后你走后，只留下星空。那一年我们望着星空。有那么多的，灿烂的梦。以为快乐会永久。像不变星空，陪着我。猎户，天狼，织女，光年外沉默。回忆，青春，梦想，何时偷偷陨落。我爱过，然后我沉默，人海里漂流。那一年我们望着星空。未来的未来。从没想过。当故事失去美梦。美梦失去线索。而我们失去联络。这一片无言无语星空。为什么静静。看我泪流。如果你在的时候。会不会伸手，拥抱我。细数繁星闪烁，细数此生奔波。原来，所有，所得，所获。不如一夜的星空。空气中的温柔，回忆你的笑容。仿佛只要伸手，就能触摸。摸不到的颜色，是否叫彩虹。看不到的拥抱，是否叫做微风。一个人，习惯一个人。这一刻独自望着星空。从前的从前，从没变过。寂寞可以是忍受，也可以是享受。享受仅有的拥有。那一年我们望着星空。有那么多的，灿烂的梦。至少回忆会永久。像不变星空，陪着我。最后只剩下星空，像不变回忆。陪着我。"
    }
    
    # 单首分析
    result = analyzer.analyze(song_data)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    
    # 批量分析
    # results = analyzer.batch_analyze(all_songs)