#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""验证输出文件的脚本"""

import json
import os

data_dir = '/Users/m1/Code/net_charts/songs/data/mayday'

# 检查全图版本
full_path = os.path.join(data_dir, 'graph_word_song_data_full.json')
lcc_path = os.path.join(data_dir, 'graph_word_song_data_full_max_graph.json')

print("=== graph_word_song_data_full.json (全图版本) ===")
with open(full_path, 'r', encoding='utf-8') as f:
    full_data = json.load(f)

for pos in ['n', 'a', 'v', 't']:
    if pos in full_data:
        nodes = len(full_data[pos]['nodes'])
        edges = len(full_data[pos]['edges'])
        print(f"  {pos}: {nodes} 节点, {edges} 条边")

print("\n=== graph_word_song_data_full_max_graph.json (LCC版本) ===")
with open(lcc_path, 'r', encoding='utf-8') as f:
    lcc_data = json.load(f)

for pos in ['n', 'a', 'v', 't']:
    if pos in lcc_data:
        nodes = len(lcc_data[pos]['nodes'])
        edges = len(lcc_data[pos]['edges'])
        print(f"  {pos}: {nodes} 节点, {edges} 条边")

print("\n=== 对比分析 ===")
for pos in ['n', 'a', 'v', 't']:
    if pos in full_data and pos in lcc_data:
        full_nodes = len(full_data[pos]['nodes'])
        lcc_nodes = len(lcc_data[pos]['nodes'])
        full_edges = len(full_data[pos]['edges'])
        lcc_edges = len(lcc_data[pos]['edges'])
        
        if full_nodes == lcc_nodes and full_edges == lcc_edges:
            print(f"  {pos}: 全图和LCC相同（整个图已是一个连通分量）")
        else:
            print(f"  {pos}: 全图 {full_nodes}节点/{full_edges}边 > LCC {lcc_nodes}节点/{lcc_edges}边")

print("\n✓ 文件验证完成")
