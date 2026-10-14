#!/usr/bin/env python
"""获取 Kijai/WanVideo_comfy 所有模型文件名"""
import json
import os
import urllib.request

all_files = []
base_url = 'https://huggingface.co/api/models/Kijai/WanVideo_comfy/tree/main'

def fetch_dir(url, depth=0):
    """递归获取目录文件"""
    if depth > 2:
        return
    try:
        with urllib.request.urlopen(url, timeout=30) as resp:
            items = json.loads(resp.read().decode())
            for item in items:
                path = item.get('path', '')
                if item['type'] == 'file':
                    if path.endswith(('.safetensors', '.gguf', '.ckpt', '.bin', '.pt')):
                        all_files.append(path)
                elif item['type'] == 'directory':
                    fetch_dir(f'{base_url}/{path}', depth + 1)
    except Exception as e:
        print(f"Error fetching {url}: {e}")

print("Fetching Kijai/WanVideo_comfy models...")
fetch_dir(base_url)

# 保存到文件
with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "samples", "kijai_all_models.txt"), 'w', encoding='utf-8') as f:
    for file in sorted(all_files):
        f.write(file + '\n')

print(f'\n=== Total models: {len(all_files)} ===\n')
for f in sorted(all_files)[:50]:
    print(f)
