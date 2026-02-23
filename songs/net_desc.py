import json
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')

def load_json(filename):
    """加载JSON文件"""
    path = os.path.join(DATA_DIR, filename)
    if not os.path.exists(path):
        print(f"Warning: JSON file not found: {path}")
        return {}
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

def extract_used_data(data, word_type):
    """从完整数据中提取只用到的字段"""
    if word_type not in data:
        return {}

    item = data[word_type]
    return {
        'basic_stats': {
            'description': item.get('basic_stats', {}).get('description', ''),
            'is_connected': item.get('basic_stats', {}).get('is_connected', True)
        },
        'degree_stats': {
            'stats_overall': {
                'description': item.get('degree_stats', {}).get('stats_overall', {}).get('description', '')
            },
            'stats_words': {
                'description': item.get('degree_stats', {}).get('stats_words', {}).get('description', '')
            },
            'stats_songs': {
                'description': item.get('degree_stats', {}).get('stats_songs', {}).get('description', '')
            }
        }
    }

def merge_network_data(file_path):
    """合并3个网络分析JSON文件,仅保留用到的数据"""
    # 加载3个源文件
    data_metrics = load_json(f'{file_path}network_metrics_data_full.json')
    data_l_path = load_json(f'{file_path}graph_longest_path_data_full.json')
    data_l_cycle = load_json(f'{file_path}graph_largest_cycle_data_full.json')

    word_types = ['n', 'v', 'a', 't']

    # 创建合并后的数据结构,仅保留用到的字段
    merged_data = {
        "network_metrics": {},
        "longest_path": {},
        "largest_cycle": {}
    }

    # 提取 network_metrics 中的用到字段
    for word_type in word_types:
        merged_data['network_metrics'][word_type] = extract_used_data(data_metrics, word_type)

    # 提取 longest_path 中的用到字段
    for word_type in word_types:
        if word_type in data_l_path:
            merged_data['longest_path'][word_type] = {
                'data_info': {
                    'description': data_l_path[word_type].get('data_info', {}).get('description', '')
                }
            }

    # 提取 largest_cycle 中的用到字段
    for word_type in word_types:
        if word_type in data_l_cycle:
            merged_data['largest_cycle'][word_type] = {
                'data_info': {
                    'description': data_l_cycle[word_type].get('data_info', {}).get('description', '')
                }
            }

    # 保存合并后的JSON - 修复路径问题
    output_filename = f'{file_path}network_analysis_data_merged.json'
    output_path = os.path.join(DATA_DIR, output_filename)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(merged_data, f, ensure_ascii=False, indent=4)

    print(f"✓ 合并完成!输出文件:{output_path}")
    print(f"✓ 原始文件总大小 → 精简后数据")
    return merged_data


if __name__ == '__main__':
    # file_path_prefix = "jaychou/"
    # file_path_prefix = "mayday/"
    # file_path_prefix = "liuyuning/"
    file_path_prefix = "liyuchun/"
    merge_network_data(file_path_prefix)
