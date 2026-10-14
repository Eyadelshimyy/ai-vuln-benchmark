#!/usr/bin/env python
"""
获取 Comfy-Org 和 GGUF 仓库所有模型文件名
目标仓库:
1. Comfy-Org (官方)
2. City96 (ComfyUI-GGUF 作者)
3. bartowski (GGUF 量化大户)
4. mradermacher (GGUF 量化大户)
"""
import json
import urllib.request
import time
import os

TARGET_REPOS = [
    # Comfy-Org 官方仓库 (前 50 个热门)
    {"id": "Comfy-Org", "type": "org", "limit": 50},
    # GGUF 量化仓库
    {"id": "City96", "type": "user", "limit": 50},
    {"id": "bartowski", "type": "user", "limit": 50},
    {"id": "mradermacher", "type": "user", "limit": 50},
    {"id": "MaziyarPanahi", "type": "user", "limit": 30},
]

all_files = {}

def get_repo_list(author, limit=50):
    """获取用户/组织的模型仓库列表"""
    url = f"https://huggingface.co/api/models?author={author}&sort=downloads&direction=-1&limit={limit}"
    try:
        with urllib.request.urlopen(url, timeout=30) as resp:
            data = json.loads(resp.read().decode())
            return [model['modelId'] for model in data]
    except Exception as e:
        print(f"Error fetching repos for {author}: {e}")
        return []

def fetch_dir(base_url, dir_path="", depth=0):
    """递归获取目录文件"""
    files = []
    if depth > 3: # 限制深度
        return files
        
    target_url = f"{base_url}/tree/main/{dir_path}" if dir_path else f"{base_url}/tree/main"
    # 处理 URL 中的空格
    target_url = target_url.replace(" ", "%20")
    
    try:
        with urllib.request.urlopen(target_url, timeout=20) as resp:
            items = json.loads(resp.read().decode())
            for item in items:
                path = item.get('path', '')
                if item['type'] == 'file':
                    ext = path.lower().split('.')[-1]
                    if ext in ['safetensors', 'gguf', 'ckpt', 'bin', 'pt', 'pth']:
                        files.append(path)
                elif item['type'] == 'directory':
                    # 递归子目录
                    sub_files = fetch_dir(base_url, path, depth + 1)
                    files.extend(sub_files)
    except Exception as e:
        # print(f"  Error scanning {target_url}: {e}")
        pass
    
    return files

def main():
    total_models = 0
    
    for target in TARGET_REPOS:
        author = target['id']
        print(f"\nScanning {author} ({target['type']})...")
        
        repos = get_repo_list(author, target['limit'])
        print(f"  Found {len(repos)} repositories")
        
        for repo_id in repos:
            print(f"    Scanning {repo_id}...", end="", flush=True)
            files = fetch_dir(f"https://huggingface.co/api/models/{repo_id}")
            if files:
                all_files[repo_id] = files
                total_models += len(files)
                print(f" {len(files)} files")
            else:
                print(" 0 files")
            time.sleep(0.1) # 避免速率限制

    # 保存结果
    output_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "samples", "comfy_gguf_models.json")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(all_files, f, indent=2, ensure_ascii=False)

    print(f"\n=== Completed! Copied {total_models} files from {len(all_files)} repos ===")
    print(f"Saved to {output_file}")

if __name__ == "__main__":
    main()
