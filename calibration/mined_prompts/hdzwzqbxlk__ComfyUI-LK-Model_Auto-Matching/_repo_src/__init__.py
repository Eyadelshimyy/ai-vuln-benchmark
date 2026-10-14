import logging

import server
from aiohttp import web
from .core.scanner import ModelScanner
from .core.matcher import ModelMatcher
from .core.searcher import ModelSearcher

logger = logging.getLogger("AutoModelMatcher")


class ErrorCode:
    VALIDATION_ERROR       = "VALIDATION_ERROR"
    MATCH_FAILED           = "MATCH_FAILED"
    SEARCH_FAILED          = "SEARCH_FAILED"
    INDEX_REFRESH_FAILED   = "INDEX_REFRESH_FAILED"
    CONFIG_SAVE_FAILED     = "CONFIG_SAVE_FAILED"
    CONFIG_VALIDATE_FAILED = "CONFIG_VALIDATE_FAILED"
    CONFIG_LOAD_FAILED     = "CONFIG_LOAD_FAILED"
    INTERNAL_ERROR         = "INTERNAL_ERROR"


def error_response(code: str, message: str, status: int = 500):
    # `error` stays a STRING for backward compat with js/auto_matcher.js:196.
    # `message` must be user-safe (no stack/trace/path).
    return web.json_response({"error": message, "code": code}, status=status)


__version__ = "3.6.2" # Zero-Hashing Bi-Directional Fast Path Alignment Indexing
__author__ = "LK"

# 初始化核心组件
scanner = ModelScanner()
matcher = ModelMatcher(scanner)
searcher = ModelSearcher()

# 注册 API 路由
@server.PromptServer.instance.routes.post("/auto-matcher/match")
async def match_models(request):
    try:
        data = await request.json()
        items = data.get("items", [])
        
        # 调用新版 matcher，传入列表
        matches = matcher.match(items)
        
        # 格式化返回结果
        results = []
        for m in matches:
            results.append({
                "id": m["id"],
                "node_type": m["node_type"],
                "widget_name": m["widget_name"], 
                "original": m["original_value"], 
                "new_value": m["matched_value"],
                "match_type": m.get("match_type", "Fuzzy"),
                "type": m.get("type", "unknown")
            })
        
        return web.json_response({"matches": results})
    except Exception as e:
        logger.exception(f"[AutoModelMatcher] API Error: {e}")
        return error_response(ErrorCode.MATCH_FAILED, "模型匹配失败，请稍后重试。")

@server.PromptServer.instance.routes.post("/auto-matcher/search")
async def search_models(request):
    try:
        data = await request.json()
        # 预期输入: {"items": [{"current": "v1.5.ckpt"}, ...], "ignore_cache": boolean} 
        items = data.get("items", [])
        ignore_cache = data.get("ignore_cache", False)
        
        # 准备并发任务
        tasks = []
        original_filenames = []
        results = []
        
        from .core.scanner import is_valid_model_file
        import asyncio
        
        # [v3.0.3] 并发搜索前，先进行本地扫描
        for item in items:
            filename = item.get("current")
            if not filename: continue
            
            # 1. 检查本地是否存在 (暴力搜索所有模型目录)
            local_path = scanner.find_local_file(filename)
            if local_path:
                # 如果本地找到了，直接返回特殊结果
                results.append({
                    "original": filename,
                    "type": item.get("type", "unknown"),
                    "result": {
                        "source": "Local Disk (Unindexed)",
                        "name": "Found Locally",
                        "filename": filename,
                        "url": "",  # 不需要下载
                        "pageUrl": "",
                        "score": 1.0,
                        "local_path": local_path,
                        "description": "File exists on disk but not in ComfyUI index. Please Refresh."
                    }
                })
                continue

            # 2. 本地没找到，加入网络搜索队列
            # Strict Filter at API level
            if is_valid_model_file(filename):
                tasks.append(searcher.search(filename, ignore_cache=ignore_cache))
                original_filenames.append(filename)

        if tasks:
            # 并发执行所有搜索
            search_results = await asyncio.gather(*tasks)
            for filename, result_list in zip(original_filenames, search_results):
                if result_list and isinstance(result_list, list) and len(result_list) > 0:
                    results.append({
                        "original": filename,
                        "type": item.get("type", "unknown"),
                        "result": result_list[0] # Unwrap list to get the best match object
                    })
        
        return web.json_response({"downloads": results})
    except Exception as e:
        logger.exception(f"[AutoModelMatcher] Search API Error: {e}")
        return error_response(ErrorCode.SEARCH_FAILED, "模型搜索失败，请检查网络后重试。")

@server.PromptServer.instance.routes.post("/auto-matcher/refresh-index")
async def refresh_index(request):
    try:
        count = scanner.scan_incremental()
        matcher.invalidate_index()
        return web.json_response({"status": "ok", "count": count})
    except Exception as e:
        logger.exception(f"[AutoModelMatcher] Index Refresh Error: {e}")
        return error_response(ErrorCode.INDEX_REFRESH_FAILED, "索引刷新失败，请稍后重试。")

@server.PromptServer.instance.routes.post("/auto-matcher/save-config")
async def save_config(request):
    try:
        data = await request.json()
        searcher.save_config(data)
        return web.json_response({"status": "ok"})
    except Exception as e:
        logger.exception(f"[AutoModelMatcher] Save Config Error: {e}")
        return error_response(ErrorCode.CONFIG_SAVE_FAILED, "配置保存失败，请稍后重试。")

@server.PromptServer.instance.routes.post("/auto-matcher/validate-config")
async def validate_config(request):
    try:
        data = await request.json()
        api_key = data.get("civitai_api_key", "")
        is_valid, msg = await searcher.validate_api_key(api_key)
        return web.json_response({"valid": is_valid, "message": msg})
    except Exception as e:
        logger.exception(f"[AutoModelMatcher] Validate Config Error: {e}")
        return error_response(ErrorCode.CONFIG_VALIDATE_FAILED, "配置校验失败，请稍后重试。")

@server.PromptServer.instance.routes.get("/auto-matcher/get-config")
async def get_config(request):
    try:
        config = searcher.get_config()
        # Return version for frontend sync
        response_data = config.copy() if config else {}
        response_data["version"] = __version__
        return web.json_response(response_data)
    except Exception as e:
        logger.exception(f"[AutoModelMatcher] Get Config Error: {e}")
        return error_response(ErrorCode.CONFIG_LOAD_FAILED, "配置读取失败，请稍后重试。")

# 插件目录配置
WEB_DIRECTORY = "./js"
NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]

logger.info("\033[34m[AutoModelMatcher] \033[0mLoaded successfully with API support.")
