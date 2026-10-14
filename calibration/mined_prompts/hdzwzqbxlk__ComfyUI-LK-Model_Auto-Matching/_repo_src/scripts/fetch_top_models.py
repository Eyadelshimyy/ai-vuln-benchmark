from curl_cffi import requests as c_requests
import sys
import os
import json

# 添加 core 到 path 以便导入 database
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(__file__)), "core"))

try:
    from database import db
except ImportError as e:
    print(f"Error importing database: {e}")
    sys.exit(1)

def fetch_civitai_models(limit=100):
    """
    从 Civitai API 获取热门模型
    """
    url = "https://civitai.com/api/v1/models"
    params = {
        "sort": "Most Downloaded",
        "limit": limit,
        "types": "Checkpoint,LORA,TextualInversion,VAE" # 获取主要类型
    }
    
    # Load config
    config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "envs", "config.json")
    api_key = ""
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
            api_key = config.get("civitai_api_key", "")
    except: pass

    # curl_cffi 自动处理大部分 User-Agent 和 TLS 指纹
    headers = {
        "Referer": "https://civitai.com/",
        "Origin": "https://civitai.com"
    }
    
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
        print(f"Using API Key: {api_key[:8]}...")
    
    print(f"Fetching top {limit} models from Civitai...")
    try:
        # 使用 chrome120 模拟浏览器指纹，绕过 Cloudflare
        response = c_requests.get(url, params=params, headers=headers, impersonate="chrome120", timeout=30)
        
        if response.status_code != 200:
            print(f"Request failed: {response.status_code} {response.text[:200]}")
            return []
            
        return response.json().get("items", [])
    except Exception as e:
        print(f"Request failed: {e}")
        return []

def populate_db(models):
    """
    将模型数据插入数据库
    """
    count = 0
    for item in models:
        name = item.get("name")
        model_type = item.get("type")
        
        # 简单的 Base Model 推断
        base_model = "Unknown"
        model_versions = item.get("modelVersions", [])
        if model_versions:
            # 取最新版本的基础模型
            latest_ver = model_versions[0]
            base_model_set = latest_ver.get("baseModel")
            if base_model_set:
                base_model = base_model_set
        
        description = f"Civitai ID: {item.get('id')}"
        
        print(f"Adding: {name} ({model_type} / {base_model})")
        
        # 1. 添加模型
        model_id = db.add_model(name, model_type, base_model, description)
        
        # 2. 添加文件哈希 (如果有)
        for version in model_versions:
             files = version.get("files", [])
             for file in files:
                 sha256 = file.get("hashes", {}).get("SHA256")
                 filename = file.get("name")
                 if sha256:
                     db.add_hash(sha256, model_id, filename, source="Civitai")
        
        # 3. 添加别名 (如果有 triggers 或者是简单的名字变体)
        # 这里暂时只添加名字的小写作为 alias
        db.add_alias(name, model_id, is_regex=False)
        
        count += 1
        
    print(f"Done. Processed {count} models.")

if __name__ == "__main__":
    models = fetch_civitai_models(limit=20) # 默认只取前20个测试
    if models:
        populate_db(models)
    else:
        print("No models fetched.")
