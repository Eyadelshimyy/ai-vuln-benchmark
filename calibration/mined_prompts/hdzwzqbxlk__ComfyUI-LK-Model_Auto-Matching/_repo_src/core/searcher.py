import asyncio
import logging
import urllib.parse
import os
import json
import re
import random
import hashlib
import time
import uuid
from curl_cffi.requests import AsyncSession
from parsel import Selector

logger = logging.getLogger(__name__)

try:
    from .utils import AdvancedTokenizer
    from .models_db_reader import find_best_match_in_db
    from .mirror import rewrite_hf_url, rewrite_modelscope_url
except ImportError:
    from utils import AdvancedTokenizer
    from models_db_reader import find_best_match_in_db
    from mirror import rewrite_hf_url, rewrite_modelscope_url

class BaseProvider:
    def __init__(self, config=None):
        self.config = config or {}
        # Update to newer impersonation to avoid blocking
        # curl_cffi supports chrome124 in newer versions, or verify installed version
        self.impersonate = "chrome124"
        # [v3.5.2] Read timeout from config, default to 20s for better stability
        self.timeout = self.config.get("network", {}).get("timeout", 20)

    def _get_headers(self, referer=None):
        headers = {
            "Accept": "application/json, text/plain, */*",
            "Accept-Encoding": "gzip, deflate, br",
            "Accept-Language": "en-US,en;q=0.9",
            # "User-Agent": ... -> REMOVED: Let curl_cffi handle UA to match TLS fingerprint
            "Connection": "keep-alive"
        }
        if referer:
            headers["Referer"] = referer
        return headers


class CivitaiHashProvider(BaseProvider):
    """
    [v3.0.1] 通过 SHA256 哈希精确匹配 Civitai 模型
    
    Civitai 官方 API: /api/v1/model-versions/by-hash/{hash}
    参考: ComfyUI-Lora-Manager 实现
    
    准确率: 100% (对于从 Civitai 下载的模型)
    """
    def __init__(self, config):
        super().__init__(config)
        self.api_url = "https://civitai.com/api/v1/model-versions/by-hash"
        self._hash_cache = {}  # 缓存已计算的哈希
    
    @staticmethod
    def calculate_sha256(file_path: str) -> str:
        """计算文件 SHA256 哈希 (支持大文件)"""
        sha256 = hashlib.sha256()
        with open(file_path, 'rb') as f:
            for chunk in iter(lambda: f.read(65536), b''):  # 64KB chunks
                sha256.update(chunk)
        return sha256.hexdigest()
    
    async def search_by_hash(self, file_path: str, original_filename: str):
        """Search Civitai by SHA256"""
        if getattr(self, "circuit_open", False):
            logger.warning(f"[CivitaiProvider] Circuit Open (Blocking skipped)")
            await asyncio.sleep(0) # Yield
            return None

        # Calculate Hash (Async)
        # ...
        
        try:
            # ... request ...
            pass
        except Exception as e:
            if "403" in str(e):
                self.error_count = getattr(self, "error_count", 0) + 1
                if self.error_count >= 3:
                    self.circuit_open = True
                    logger.warning(f"[CivitaiProvider] 403 limit reached. Disabling Civitai for this session.")
            return None
        """
        通过文件哈希精确匹配 Civitai 模型
        
        Args:
            file_path: 本地文件完整路径
            original_filename: 原始文件名
            
        Returns:
            list: 匹配结果列表 (0或1个结果)
        """
        results = []
        
        if not os.path.exists(file_path):
            logger.warning(f"[CivitaiHash] File not found: {file_path}")
            return results
        
        try:
            # 检查缓存
            if file_path in self._hash_cache:
                file_hash = self._hash_cache[file_path]
            else:
                logger.debug(f"[CivitaiHash] Calculating SHA256 for: {original_filename}")
                # [Optimization] Run hashing in a separate thread to avoid blocking the Event Loop
                file_hash = await asyncio.to_thread(self.calculate_sha256, file_path)
                self._hash_cache[file_path] = file_hash
                logger.debug(f"[CivitaiHash] Hash: {file_hash[:16]}...")
            
            # [v3.0.3] 调用 Civitai API - 添加 Origin/Referer 绕过 Cloudflare
            headers = self._get_headers("https://civitai.com")
            headers["Origin"] = "https://civitai.com"
            headers["Referer"] = "https://civitai.com/"
            
            token = self.config.get("civitai_api_key")
            if token:
                headers["Authorization"] = f"Bearer {token}"
                logger.debug(f"[CivitaiHash] Using API Key: {token[:8]}...")
            else:
                logger.warning(f"[CivitaiHash] Warning: No API Key configured")

            
            url = f"{self.api_url}/{file_hash}"
            
            async with AsyncSession(impersonate=self.impersonate, headers=headers, timeout=self.timeout) as session:
                response = await session.get(url)
                
                if response.status_code == 200:
                    try:
                        data = response.json()
                        
                        model_id = data.get("modelId")
                        model_name = data.get("model", {}).get("name", "Unknown")
                        version_name = data.get("name", "")
                        
                        # 构建下载 URL
                        download_url = data.get("downloadUrl", "")
                        if not download_url:
                            # 回退到标准模型页面
                            download_url = f"https://civitai.com/models/{model_id}"
                        
                        results.append({
                            "source": "Civitai (Hash Match)",
                            "name": f"{model_name} - {version_name}",
                            "filename": original_filename,
                            "url": download_url,
                            "pageUrl": f"https://civitai.com/models/{model_id}",
                            "score": 1.0,  # 100% 确定匹配
                            "hash_match": True
                        })
                        
                        logger.info(f"[CivitaiHash] ✓ Exact match found: {model_name}")
                        
                    except Exception as e:
                        logger.exception(f"[CivitaiHash] Parse error: {e}")
                        
                elif response.status_code == 404:
                    logger.info(f"[CivitaiHash] No match for hash (model may not be from Civitai)")
                else:
                    logger.warning(f"[CivitaiHash] API returned {response.status_code}")
                    
        except Exception as e:
            logger.exception(f"[CivitaiHash] Error: {e}")
        
        return results


class CivitaiProvider(BaseProvider):

    def __init__(self, config):
        super().__init__(config)
        self.api_url = "https://civitai.com/api/v1/models"
    
    async def search(self, query, original_filename):
        results = []
        try:
            logger.debug(f"[CivitaiProvider] Searching API for: {query}")
            headers = self._get_headers("https://civitai.com")
            
            # [Fix] Add Origin/Referer to satisfy Cloudflare/Anti-bot checks
            headers["Origin"] = "https://civitai.com"
            headers["Referer"] = "https://civitai.com/"
            
            token = self.config.get("civitai_api_key")
            if token:
                headers["Authorization"] = f"Bearer {token}"
                # Log that key is being used (masked)
                logger.debug(f"[CivitaiProvider] Authenticated search (Key ends found)")

            encoded_query = urllib.parse.quote(query)
            # Fetch more results to increase hit rate
            url = f"{self.api_url}?query={encoded_query}&limit=20"
            
            async with AsyncSession(impersonate=self.impersonate, headers=headers, timeout=self.timeout) as session:
                response = await session.get(url)
                if response.status_code != 200: 
                    logger.error(f"[CivitaiProvider] API Error {response.status_code} (Check API Key or Network)")
                    # Detailed debug for 403
                    if response.status_code == 403:
                        logger.warning(f"[CivitaiProvider] 403 Forbidden. Headers sent: {headers.keys()}")
                    return []
                
                try:
                    data = response.json()
                except: return []

                items = data.get("items", [])
                original_lower = original_filename.lower()
                
                for item in items:
                    model_name = item.get("name", "")
                    model_id = item.get("id")
                    
                    for version in item.get("modelVersions", []):
                        ver_name = version.get("name", "")
                        ver_id = version.get("id")
                        
                        for file_info in version.get("files", []):
                            fname = file_info.get("name", "")
                            if not fname: continue
                            
                            # Scoring
                            fname_base = os.path.splitext(fname)[0].lower()
                            file_score = AdvancedTokenizer.calculate_similarity(original_lower, fname_base)
                            
                            # Strict exclusion
                            if file_score <= 0.05: continue
                            
                            combined_name = f"{model_name} {ver_name}"
                            name_score = AdvancedTokenizer.calculate_similarity(original_lower, combined_name.lower())
                            
                            # Final weighted score
                            final_score = max(file_score, (file_score * 0.7 + name_score * 0.3))
                            
                            if final_score > 0.35:
                                results.append({
                                    "source": "Civitai (Native)",
                                    "name": f"{model_name} - {ver_name}",
                                    "filename": fname,
                                    "url": file_info.get("downloadUrl"),
                                    "pageUrl": f"https://civitai.com/models/{model_id}?modelVersionId={ver_id}",
                                    "score": final_score
                                })
        except Exception as e:
            logger.exception(f"[CivitaiProvider] Error: {e}")
        return results

class HuggingFaceProvider(BaseProvider):
    def __init__(self, config):
        super().__init__(config)
        self.api_url = "https://huggingface.co/api/models"

    async def search(self, query, original_filename):
        results = []
        try:
            logger.debug(f"[HFProvider] Searching API for: {query}")
            headers = self._get_headers("https://huggingface.co")
            encoded_query = urllib.parse.quote(query)
            url = f"{self.api_url}?search={encoded_query}&limit=20"
            
            async with AsyncSession(impersonate=self.impersonate, headers=headers, timeout=self.timeout) as session:
                response = await session.get(url)
                if response.status_code != 200: return []
                
                try:
                    data = response.json()
                except: return []
                
                original_lower = original_filename.lower()
                
                for repo in data:
                    model_id = repo.get("modelId", "")
                    if not model_id: continue
                    
                    repo_name_clean = model_id.split("/")[-1]
                    
                    score = AdvancedTokenizer.calculate_similarity(original_lower, repo_name_clean.lower())
                    full_score = AdvancedTokenizer.calculate_similarity(original_lower, model_id.lower().replace("/", " "))
                    final_score = max(score, full_score)
                    
                    if final_score > 0.35:
                        results.append({
                            "source": "HuggingFace",
                            "name": model_id,
                            "filename": f"{repo_name_clean}.safetensors", 
                            "url": f"https://huggingface.co/{model_id}/tree/main",
                            "pageUrl": f"https://huggingface.co/{model_id}",
                            "score": final_score
                        })
        except Exception as e:
            logger.exception(f"[HFProvider] Error: {e}")
        return results

class HuggingFaceFileSearchProvider(BaseProvider):
    """
    [v3.3.0] 高性能 HuggingFace 文件搜索
    
    优化策略：
    1. 并发目录遍历 (asyncio.gather)
    2. 智能剪枝 (仅扫描相关目录)
    3. 仓库结构缓存 (5分钟TTL)
    4. 早停机制 (找到匹配立即返回)
    """
    
    # 类级别缓存 (所有实例共享)
    _tree_cache = {}  # {model_id: {"tree": {...}, "ts": timestamp}}
    CACHE_TTL = 300   # 5 分钟
    
    def __init__(self, config):
        super().__init__(config)
        self.api_url = "https://huggingface.co/api/models"
        self.timeout = 20 # Deep camouflage needs more time
        self.impersonate = "chrome124" # Update to newer browser
        
    def _get_weighted_tokens(self, filename):
        """[v3.4.1] 提取带权重的 tokens"""
        base_name = os.path.splitext(filename)[0]
        
        # 1. CamelCase & SnakeCase Split
        base_name = re.sub(r'(?i)Wan21', 'Wan 2.1', base_name) # Fix aniWan21 -> Wan 2.1
        base_name = re.sub(r'([a-z])([A-Z])', r'\1 \2', base_name)
        # 1.1 Letter-Number Split (Fix Wan2.1 -> Wan 2.1, combined with Model Size Split)
        # Handle 2114B -> 21 14B, 307B -> 30 7B
        base_name = re.sub(r'(\d+)(14[bB]|7[bB]|8[bB]|72[bB]|32[bB]|1\.5[bB])', r'\1 \2', base_name)
        # Handle 14BFp -> 14B Fp
        base_name = re.sub(r'(14[bB]|7[bB]|8[bB]|72[bB]|32[bB]|1\.5[bB])([a-zA-Z])', r'\1 \2', base_name)
        
        base_name = re.sub(r'([a-z])(?<![vit])(\d)', r'\1 \2', base_name)
        base_name = re.sub(r'(\d)(?!(?i:[bkmgv]))([a-zA-Z])', r'\1 \2', base_name)
        
        # 2. Split
        raw_tokens = re.split(r'[-_\s]+', base_name)
        
        weighted_tokens = []
        for t in raw_tokens:
            if not t: continue
            
            # Handle dots (2.1 vs style.lora)
            sub_tokens = []
            if re.match(r'^v?[\d]+\.[\d]+$', t): 
                sub_tokens = [t]
            else:
                sub_tokens = t.split('.')
                
            for token in sub_tokens:
                if not token or (len(token) < 2 and not token.isdigit()): continue
                
                token_lower = token.lower()
                weight = 5
                
                # Weighting Rules
                if token_lower in {'kijai', 'comfy', 'org', 'city96', 'bartowski', 'maziyarpanahi', 'mradermacher', 'wan-ai', 'nvidia'}:
                    weight = 10
                elif token_lower in {'wan', 'flux', 'qwen', 'sdxl', 'pony', 'hunyuan', 'lumina', 'cosmos', 'deepseek', 'llama', 'mistral', 'gemma'}:
                    weight = 8
                elif re.match(r'^v?[\d\.]+$', token_lower) or token_lower in {'xl', 'turbo', 'dev', 'schnell'}:
                    weight = 6
                elif re.match(r'^\d+b$', token_lower):
                    weight = 7
                elif token_lower in {'gguf', 'lora', 't2v', 'i2v', 'vae', 'controlnet', 'inpaint'}:
                    weight = 4
                elif token_lower in {'bf16', 'fp16', 'fp8', 'int8', 'rank', 'average', 'pruned', 'full', 'merged', 'safetensors'}:
                    weight = 1
                elif token_lower.isdigit():
                    weight = 3 
                    
                weighted_tokens.append({'token': token_lower, 'weight': weight})
                
        return weighted_tokens

    def _extract_keywords(self, filename):
        """[v3.4.1] 返回高权重关键词列表 (用于搜索)"""
        weighted = self._get_weighted_tokens(filename)
        # Sort by weight desc
        weighted.sort(key=lambda x: x['weight'], reverse=True)
        return [item['token'] for item in weighted if item['weight'] > 1] # Remove Noise
        return [item['token'] for item in weighted_tokens if item['weight'] > 1]
        
    def _get_priority_authors(self):
        """[Bugfix] 读取配置中的高优先级 HF 组织/作者列表，用于优先搜索官方镜像。

        配置来源：``searcher.priority_organizations.huggingface``（统一配置），
        向后兼容旧的 ``searcher.api.huggingface.priority_authors``，最终回落默认值。
        """
        authors = self.config.get("searcher", {}).get("priority_organizations", {}).get("huggingface", [])
        if not authors:
            authors = self.config.get("searcher", {}).get("api", {}).get("huggingface", {}).get("priority_authors", [])
        if not authors:
            try:
                from .config import get_all_config
                authors = get_all_config().get("searcher", {}).get("priority_organizations", {}).get("huggingface", [])
            except Exception:
                pass
        if not authors:
            try:
                from .config import get_all_config
                authors = get_all_config().get("searcher", {}).get("api", {}).get("huggingface", {}).get("priority_authors", [])
            except Exception:
                pass
        return authors or ["Comfy-Org", "unsloth"]

    def _build_search_queries(self, keywords, max_tokens=2):
        """[Bugfix] 构建聚焦的 HF 仓库搜索词（默认取前 2 个高权重词）。

        关键修复：之前用全部高权重词（如 ``minimax music dit``）去 ``author=Comfy-Org``
        检索会返回空——HF 全文检索对过长的多词查询很挑剔。聚焦到模型族/版本级
        （``minimax music``）才能稳定命中官方组织仓库（如 ``Comfy-Org/MiniMax-Music-3``）。

        过滤掉孤立的纯整数 token（如 music3 -> '3'、fp16 -> '16'），这些整数权重高
        但对召回有害，会导致官方镜像被淹没。
        """
        # 取前 max_tokens 个高权重词（keywords 已按权重降序），剔除纯整数
        cleaned = [k for k in keywords if not re.match(r'^\d+$', k)][:max_tokens]
        # 保底：如果清洗后太少，回退到原始前 max_tokens
        if len(cleaned) < 1:
            cleaned = keywords[:max_tokens]
        return cleaned

    async def _discover_repos(self, session, keywords):
        """[v3.4.0] 动态仓库发现 (根据高权重关键词搜索 HF 仓库)

        [Bugfix] 先搜索 Comfy-Org / unsloth 等高优先级组织命名空间，
        再执行通用关键词搜索。避免官方镜像因下载量低而被原始仓库淹没。
        """
        # 聚焦查询：模型族/版本级（默认前 2 高权重词），确保 official 组织检索稳定命中。
        # 例如 minimax_music3_dit -> ['minimax','music']，用 'minimax music' 检索
        # author=Comfy-Org 能命中 Comfy-Org/MiniMax-Music-3；而 'minimax music dit' 会返回空。
        priority_kw = self._build_search_queries(keywords, 2)
        priority_query = " ".join(priority_kw)
        # 通用检索查询（稍宽，取前 3 高权重词，仍过滤纯整数）
        general_kw = self._build_search_queries(keywords, 3)
        general_query = " ".join(general_kw)

        # 特殊处理 LoRA
        is_lora = any('lora' in k.lower() for k in keywords)

        logger.debug(f"[SmartDiscovery] priority_query='{priority_query}' general_query='{general_query}'")

        discovered_repos = []

        # 1) 高优先级组织命名空间搜索（Comfy-Org / unsloth 等官方镜像）
        priority_authors = self._get_priority_authors()
        for author in priority_authors:
            found = []
            # 先用聚焦 top-2 查询；若某组织无命中，退化为单族词（top-1）再试一次
            for q in [priority_query, priority_kw[0] if priority_kw else ""]:
                if not q or q in ("", None):
                    continue
                try:
                    url = f"{self.api_url}?author={urllib.parse.quote(author)}&search={urllib.parse.quote(q)}&limit=8"
                    resp = await session.get(url)
                    if resp.status_code != 200:
                        continue
                    items = resp.json()
                    for item in items:
                        repo_id = item.get("modelId")
                        if not repo_id: continue
                        if any('gguf' in k.lower() for k in keywords) and 'gguf' not in repo_id.lower():
                            continue
                        if repo_id not in discovered_repos and repo_id not in found:
                            found.append(repo_id)
                    if found:
                        break  # 聚焦查询已命中，无需再退化
                except Exception as e:
                    logger.debug(f"[SmartDiscovery] Author search '{author}' q='{q}' failed: {e}")
            discovered_repos.extend(found)

        # 2) 通用关键词搜索补全（按下载量排序）
        try:
            url = f"{self.api_url}?search={urllib.parse.quote(general_query)}&sort=downloads&direction=-1&limit=8"
            resp = await session.get(url)
            if resp.status_code == 200:
                items = resp.json()
                for item in items:
                    repo_id = item.get("modelId")
                    if not repo_id: continue
                    if any('gguf' in k.lower() for k in keywords) and 'gguf' not in repo_id.lower():
                        continue
                    if repo_id not in discovered_repos:
                        discovered_repos.append(repo_id)
        except Exception as e:
            logger.exception(f"[SmartDiscovery] Error: {e}")

        logger.info(f"[SmartDiscovery] Found {len(discovered_repos)} candidates: {discovered_repos}")
        return discovered_repos

    async def search(self, query, original_filename):
        """[v3.4.0] 智能混合搜索入口"""
        import time
        start_time = time.time()
        results = []
        
        try:
            # [v3.5.1] DB lookups are now handled centrally in ModelSearcher.search
            # to allow cross-provider resolution (ModelScope priority).
            # No longer doing early return here.
            pass
            
            # 2. 提取加权关键词
            keywords = self._extract_keywords(original_filename)
            if not keywords:
                return []
            
            headers = self._get_headers("https://huggingface.co")
            async with AsyncSession(impersonate=self.impersonate, headers=headers, timeout=self.timeout) as session:
                
                # 3. 动态发现仓库 (不再使用硬编码列表)
                target_repos = await self._discover_repos(session, keywords)
                
                # [Fallback] 如果没搜到，尝试只用前两个词
                if not target_repos and len(keywords) > 2:
                     target_repos = await self._discover_repos(session, keywords[:2])
                
                # 4. 并发深度扫描 (Deep Scan & Fuzzy Match)
                tasks = []
                for repo_id in target_repos:
                    tasks.append(self._scan_repo_concurrent(session, repo_id, keywords, original_filename))
                
                if tasks:
                    for coro in asyncio.as_completed(tasks):
                        try:
                            result = await coro
                            if result and result.get("score", 0) >= 0.9:
                                elapsed = time.time() - start_time
                                logger.info(f"[SmartMatch] Found high confidence match in {elapsed:.2f}s")
                                return [result]
                            elif result:
                                results.append(result)
                        except Exception as e:
                            logger.exception(f"[SmartMatch] Task error: {e}")
                            
        except Exception as e:
            logger.exception(f"[SearchError] {e}")
            
        elapsed = time.time() - start_time
        logger.info(f"[Search] Completed in {elapsed:.2f}s, found {len(results)} results")
        return sorted(results, key=lambda x: x.get("score", 0), reverse=True)[:3]
    
    async def _scan_repo_concurrent(self, session, model_id, keywords, original_filename):
        """并发扫描仓库目录"""
        import time
        
        try:
            # 检查缓存
            cache_key = model_id
            now = time.time()
            
            if cache_key in self._tree_cache:
                cached = self._tree_cache[cache_key]
                if now - cached["ts"] < self.CACHE_TTL:
                    tree = cached["tree"]
                    return self._search_in_tree(tree, model_id, keywords, original_filename)
            
            # 获取根目录
            tree_url = f"https://huggingface.co/api/models/{model_id}/tree/main"
            resp = await session.get(tree_url)
            if resp.status_code != 200:
                return None
            
            root_items = resp.json()
            
            # 构建树结构 (并发获取子目录)
            tree = {"files": [], "dirs": {}}
            dir_tasks = []
            
            for item in root_items:
                item_type = item.get("type", "")
                item_path = item.get("path", "")
                
                if item_type == "file":
                    tree["files"].append(item_path)
                elif item_type == "directory":
                    # 智能剪枝：只扫描与关键词相关的目录
                    dir_lower = item_path.lower()
                    should_scan = any(kw in dir_lower for kw in keywords)
                    
                    # [v3.3.2] 扩充常见目录名 (基于 Kijai/WanVideo_comfy + Comfy-Org 结构)
                    common_dirs = {
                        # 通用模型目录
                        'lora', 'loras', 'models', 'checkpoints', 'weights', 'unet',
                        'vae', 'clip', 'controlnet', 'embeddings',
                        # Comfy-Org 三级目录结构
                        'split_files', 'diffusion_models', 'text_encoders', 
                        'clip_vision', 'model_patches',
                        # Kijai/WanVideo_comfy 子目录
                        'infinitetalk', 'lightx2v', 'qwen', 'scail', 'wan22-turbo',
                        'fun', 'fastwanm', 'pusa', 'lynx', 'skyreels', 'humo',
                        'bindweave', 'camclonemaster', 'chronoedit', 'echoshot',
                        'fantasyportrait', 'flashvsr', 'kaleido', 'longxie2',
                        'mtvcrafter', 'onetoallanimation', 'ovi', 'steadydancer',
                        'unilumos', 'video-as-prompt', 'wanmove', 'wan22_funreward',
                    }
                    if any(cd in dir_lower for cd in common_dirs):
                        should_scan = True
                    
                    if should_scan:
                        dir_tasks.append(self._get_dir_files(session, model_id, item_path))
            
            # 并发获取所有相关子目录
            if dir_tasks:
                dir_results = await asyncio.gather(*dir_tasks, return_exceptions=True)
                for i, result in enumerate(dir_results):
                    if isinstance(result, dict):
                        dir_path = list(result.keys())[0] if result else None
                        if dir_path:
                            tree["dirs"][dir_path] = result[dir_path]
            
            # 缓存树结构
            self._tree_cache[cache_key] = {"tree": tree, "ts": now}
            
            # 在树中搜索
            return self._search_in_tree(tree, model_id, keywords, original_filename)
            
        except Exception as e:
            logger.exception(f"[HFOptimized] Repo scan error for {model_id}: {e}")
            return None
    
    async def _get_dir_files(self, session, model_id, dir_path, depth=0):
        """[v3.3.2] 递归获取目录文件 (最多3层，支持 Comfy-Org 三级目录)"""
        result = {dir_path: {"files": [], "dirs": {}}}
        
        if depth >= 3:  # [v3.3.2] 增加到3层以支持 Comfy-Org
            return result
        
        try:
            url = f"https://huggingface.co/api/models/{model_id}/tree/main/{dir_path}"
            resp = await session.get(url)
            if resp.status_code != 200:
                return result
            
            items = resp.json()
            sub_dir_tasks = []
            
            for item in items:
                item_type = item.get("type", "")
                item_path = item.get("path", "")
                
                if item_type == "file":
                    result[dir_path]["files"].append(item_path)
                elif item_type == "directory" and depth < 2:  # [v3.3.2] 递归2层
                    sub_dir_tasks.append(self._get_dir_files(session, model_id, item_path, depth + 1))
            
            if sub_dir_tasks:
                sub_results = await asyncio.gather(*sub_dir_tasks, return_exceptions=True)
                for sr in sub_results:
                    if isinstance(sr, dict):
                        result[dir_path]["dirs"].update(sr)
                        
        except Exception as e:
            logger.exception(f"[HFOptimized] Dir error {dir_path}: {e}")
        
        return result
    
    def _search_in_tree(self, tree, model_id, keywords, original_filename):
        """在缓存的树结构中搜索文件"""
        original_lower = original_filename.lower()
        original_base = os.path.splitext(original_filename)[0].lower()
        
        # 搜索根目录文件
        for file_path in tree.get("files", []):
            if self._is_match(file_path, original_lower, original_base, repo_id=model_id):
                return self._build_result(model_id, file_path, 0.98)
        
        # 搜索子目录
        for dir_path, dir_content in tree.get("dirs", {}).items():
            for file_path in dir_content.get("files", []):
                if self._is_match(file_path, original_lower, original_base, repo_id=model_id):
                    return self._build_result(model_id, file_path, 0.95)
            
            # 搜索嵌套子目录
            for sub_dir, sub_content in dir_content.get("dirs", {}).items():
                if isinstance(sub_content, dict):
                    for file_path in sub_content.get("files", []):
                        if self._is_match(file_path, original_lower, original_base, repo_id=model_id):
                            return self._build_result(model_id, file_path, 0.92)
        
        return None
    
    def _is_match(self, file_path, original_lower, original_base, repo_id=None):
        """[v3.7.0] 精准加权匹配 (Precision Weighted Intersection)

        修复:
        - Type 冲突硬门控（T2V↔I2V 等）在 fuzzy short-circuit 之前执行
        - Repo 上下文不再直接并入目标 token 集，改为"文件必须有最低核心重叠 + repo 弥补"
        - Fuzzy short-circuit 阈值收紧，防止单字符差异（i2v↔t2v）误判
        """
        file_base = os.path.splitext(os.path.basename(file_path))[0].lower()

        # ── 0. 双方 token 提取（文件级 & 可选 repo 级）──
        tags_source = self._get_weighted_tokens(original_base)
        tags_target_file = self._get_weighted_tokens(file_base)

        tags_repo = []
        if repo_id:
            repo_name = repo_id.split('/')[-1]
            tags_repo = self._get_weighted_tokens(repo_name)

        # 核心词 weight >= 6（模型族/版本/尺寸）
        core_source = {t['token'] for t in tags_source if t['weight'] >= 6}
        core_file   = {t['token'] for t in tags_target_file if t['weight'] >= 6}
        core_repo   = {t['token'] for t in tags_repo if t['weight'] >= 6}

        # 类型词 weight == 4（t2v / i2v / lora / controlnet / inpaint / vae）
        # 注意：_get_weighted_tokens 对 i2v/t2v 等混合编码提取不稳定，故 type gate 用独立正则
        type_source = {t['token'] for t in tags_source if t['weight'] == 4}
        type_file   = {t['token'] for t in tags_target_file if t['weight'] == 4}

        # ── 1. Type 冲突 HARD gate（在任何 fuzzy 匹配之前）──
        # 使用直接子串正则检测（不依赖 _get_weighted_tokens 的类型词提取）
        _TYPE_REGEX_MAP = {
            't2v': re.compile(r't2v|text.?to.?video'),
            'i2v': re.compile(r'i2v|image.?to.?video'),
            'lora': re.compile(r'lora(?![a-z])'),       # 避免 "loral" 误匹配
            'controlnet': re.compile(r'control[_-]?net'),
            'inpaint': re.compile(r'inpaint'),
            'vae': re.compile(r'vae(?![a-z])'),
        }
        def _detect_types(text, regex_map):
            return {key for key, pat in regex_map.items() if pat.search(text)}

        raw_type_source = _detect_types(original_base, _TYPE_REGEX_MAP)
        raw_type_file   = _detect_types(file_base, _TYPE_REGEX_MAP)

        # 合并 token-based 和 regex-based 类型检测结果（任一有信号即参与判定）
        all_type_source = type_source | raw_type_source
        all_type_file   = type_file | raw_type_file

        if all_type_source and all_type_file:
            _TYPE_CONFLICT_PAIRS = [
                {'t2v', 'i2v'},
                {'lora', 'controlnet'},
            ]
            for pair in _TYPE_CONFLICT_PAIRS:
                src_hit = bool(all_type_source & pair)
                tgt_hit = bool(all_type_file & pair)
                if src_hit and tgt_hit and not (all_type_source & all_type_file):
                    logger.debug(
                        f"  [TypeConflict] '{file_base}' vs '{original_base}' "
                        f"-> rejected (source={all_type_source}, target={all_type_file})"
                    )
                    return False

        # ── 2. Fuzzy 快速筛选（受 type gate 保护后）──
        p_score, t_score = 0, 0
        try:
            from rapidfuzz import fuzz
            p_score = fuzz.partial_ratio(original_base, file_base)
            t_score = fuzz.token_set_ratio(original_base, file_base)

            if p_score > 50 or t_score > 50:
                logger.debug(f"  [AlgoDebug] '{file_base}' vs '{original_base}' -> P:{p_score}, T:{t_score}")

            # 收紧短路阈值：要求近乎完全匹配才跳过加权检查
            # （旧阈值 90/95 会把 Wan-I2V-14B <-> Wan-T2V-14B 放行）
            if p_score >= 95 and t_score >= 98:
                return True

        except ImportError:
            pass

        # ── 3. 加权核心词匹配（主路径）──
        if not core_source:
            return p_score >= 80

        # 3a. 文件自身核心词必须覆盖全部源核心词
        missing_from_file = core_source - core_file
        if not missing_from_file:
            # 文件包含所有核心词 -> 精确匹配（type 已在 step 1 检过）
            return True

        # 3b. 文件缺核心词 -> repo 可弥补，但要求文件至少有 50% 重叠
        # （防止同仓库内完全不相关的文件被 repo 名"带进去"）
        if repo_id and core_repo:
            missing_with_repo = core_source - core_file - core_repo
            if not missing_with_repo:
                overlap_ratio = len(core_source & core_file) / len(core_source)
                if overlap_ratio >= 0.5:
                    return True

        return False
    
    def _build_result(self, model_id, file_path, score):
        """构建结果对象"""
        return {
            "source": "HuggingFace (Exact File)",
            "name": model_id,
            "filename": file_path,
            "url": rewrite_hf_url(f"https://huggingface.co/{model_id}/resolve/main/{file_path}"),
            "pageUrl": f"https://huggingface.co/{model_id}/tree/main",
            "score": score
        }




class ModelScopeFileSearchProvider(BaseProvider):
    """
    [v3.5.0] ModelScope File Search Provider (Reverse-Engineered)
    
    Features:
    1. Uses internal API /api/v1/models/{id}/repo/files for file listing
    2. Deep Camouflage with random User-Agent and Referer
    3. Direct Download Link Generation (faster domestic CDN)
    4. Caching support for repo trees
    """
    
    _tree_cache = {} # {repo_id: {"files": [...], "ts": timestamp}}
    CACHE_TTL = 300
    
    # ModelScope 常见仓库映射 (用于关键词增强)。
    # 注意：Comfy-Org / unsloth 官方镜像优先通过 priority_organizations 机制
    # （见 _get_priority_orgs + search 中的组织优先检索）实现，不再硬编码 per-model 仓库，
    # 避免把"其他组织放出的模型"误写成固定仓库 ID。
    PRIORITY_REPOS = {
        'wan': ['Wan-AI/Wan2.1-T2V-14B', 'Wan-AI/Wan2.1-I2V-14B-480P', 'Wuli001/WAN-MoE'],
        'qwen': ['Qwen/Qwen2.5-VL-7B-Instruct', 'Qwen/Qwen-VL'],
        'flux': ['AI-ModelScope/FLUX.1-dev', 'AI-ModelScope/FLUX.1-schnell'],
        'sd': ['AI-ModelScope/stable-diffusion-v1-5', 'AI-ModelScope/stable-diffusion-xl-base-1.0'],
        'hunyuan': ['Tencent-Hunyuan/HunyuanVideo'],
    }
    
    def __init__(self, config):
        super().__init__(config)
        self.api_url = "https://modelscope.cn/api/v1"
        self.timeout = 25 # Increased timeout for domestic network stability
        self.impersonate = "chrome124"

    def _get_headers(self, referer=None):
        headers = super()._get_headers(referer)
        # ModelScope specific headers
        headers["Origin"] = "https://modelscope.cn"
        if referer:
            headers["Referer"] = referer
        return headers

    def _get_priority_orgs(self):
        """[Bugfix] 读取配置中的高优先级 ModelScope 组织列表（Comfy-Org / unsloth 等官方镜像）。

        与 HuggingFace 端 ``_get_priority_authors`` 对称，统一走
        ``searcher.priority_organizations.modelscope`` 配置。
        """
        orgs = self.config.get("searcher", {}).get("priority_organizations", {}).get("modelscope", [])
        if not orgs:
            try:
                from .config import get_all_config
                orgs = get_all_config().get("searcher", {}).get("priority_organizations", {}).get("modelscope", [])
            except Exception:
                pass
        return orgs or ["Comfy-Org", "unsloth"]

    async def search(self, query, original_filename):
        """
        Implementation Strategy:
        1. Search for repositories using the general search API
        2. For top candidates, fetch file lists using the RE API
        3. Match files against original_filename
        """
        results = []
        original_lower = original_filename.lower()
        original_base = os.path.splitext(original_filename)[0].lower()
        
        # 1. Search for Repositories
        # [v3.3.2] 智能搜索词生成
        search_terms = [query]
        
        # 添加中文搜索词
        import re as re_module
        chinese_chars = re_module.findall(r'[\u4e00-\u9fff]+', original_filename)
        if chinese_chars:
            search_terms.insert(0, ''.join(chinese_chars))  # 中文优先
        
        # 添加模型系列关键词 & 优先仓库
        priority_target_repos = []
        for key, repos in self.PRIORITY_REPOS.items():
            if key in original_lower:
                search_terms.append(key)
                priority_target_repos.extend(repos)
        
        # Deduplicate terms
        search_terms = list(dict.fromkeys(search_terms))
        
        logger.debug(f"[ModelScope] Searching Repos for: {search_terms[:2]}")
        if priority_target_repos:
             logger.debug(f"[ModelScope] Adding Priority Repos: {priority_target_repos}")
        
        search_url = f"{self.api_url}/dolphin/models"
        headers = self._get_headers(referer="https://modelscope.cn/models")
        headers["Content-Type"] = "application/json"
        
        target_repos = []
        
        try:
            async with AsyncSession(impersonate=self.impersonate, headers=headers, timeout=self.timeout) as session:
                
                # Try multiple search terms until we find some repos
                for term in search_terms[:2]:
                    search_payload = {
                        "PageSize": 10,
                        "PageNumber": 1,
                        "SearchText": term,
                        "Sort": {"SortBy": "Default"}
                    }
                    
                    try:
                        # [v3.5.0] Used POST for search (PUT fallback)
                        resp = await session.post(search_url, json=search_payload)
                        if resp.status_code != 200: 
                            resp = await session.put(search_url, json=search_payload)
                            if resp.status_code != 200: continue
                        
                        data = resp.json()
                        if not data.get("Success"): continue
                        
                        models = data.get("Data", {}).get("Model", {}).get("Models", [])
                        for m in models:
                            owner = m.get("Path")
                            name = m.get("Name")
                            if owner and name:
                                # ModelScope API returns Path=Owner, Name=RepoName
                                repo_id = f"{owner}/{name}"
                                if repo_id not in target_repos:
                                    target_repos.append(repo_id)
                        
                        if target_repos: break # Found something using this term
                        
                    except Exception as exc:
                        logger.exception(f"[ModelScope] Search term '{term}' failed: {exc}")
                        continue

                # [v3.5.1] Merge Priority Repos (High Priority first)
                # Ensure priority repos are at the front
                # [Bugfix] 官方组织优先检索：对每个 priority org 做定向搜索，
                # 只保留 Path == org 的仓库，确保 Comfy-Org / unsloth 镜像被优先发现
                # （即使通用搜索将其淹没）。随后由 _apply_namespace_preference 加权。
                priority_org_repos = []
                for org in self._get_priority_orgs():
                    try:
                        payload = {
                            "PageSize": 5,
                            "PageNumber": 1,
                            "SearchText": query,
                            "Sort": {"SortBy": "Default"}
                        }
                        resp = await session.post(search_url, json=payload)
                        if resp.status_code != 200:
                            resp = await session.put(search_url, json=payload)
                            if resp.status_code != 200: continue
                        data = resp.json()
                        if not data.get("Success"): continue
                        for m in data.get("Data", {}).get("Model", {}).get("Models", []):
                            owner = m.get("Path")
                            name = m.get("Name")
                            if owner == org and name:
                                repo_id = f"{owner}/{name}"
                                if repo_id not in priority_org_repos:
                                    priority_org_repos.append(repo_id)
                    except Exception as exc:
                        logger.debug(f"[ModelScope] Priority org '{org}' search failed: {exc}")

                # 组装：官方组织优先，其次 PRIORITY_REPOS 关键词命中，最后通用搜索
                final_repos = []
                for pr in priority_org_repos:
                    if pr not in final_repos:
                        final_repos.append(pr)
                for pr in priority_target_repos:
                    if pr not in final_repos:
                        final_repos.append(pr)
                for tr in target_repos:
                    if tr not in final_repos:
                        final_repos.append(tr)
                
                # 2. Iterate Repos and Fetch Files
                tasks = []
                # Scan top 5 unique repos
                for repo_id in final_repos[:5]:
                    tasks.append(self._scan_repo_files(session, repo_id, original_filename, original_base))
                
                if tasks:
                    start_time = time.time()
                    repo_results = await asyncio.gather(*tasks)
                    elapsed = time.time() - start_time
                    logger.info(f"[ModelScope] Scanned {len(tasks)} repos in {elapsed:.2f}s")
                    
                    for res_list in repo_results:
                        if res_list:
                            results.extend(res_list)
                            
        except Exception as e:
            logger.exception(f"[ModelScope] Error: {e}")
            
        return sorted(results, key=lambda x: x.get("score", 0), reverse=True)[:5]

    async def _scan_repo_files(self, session, repo_id, original_filename, original_base):
        """Fetch files for a specific repo using hidden API"""
        results = []
        
        # Check Cache
        now = time.time()
        files = []
        if repo_id in self._tree_cache:
            cache = self._tree_cache[repo_id]
            if now - cache["ts"] < self.CACHE_TTL:
                files = cache["files"]
            else:
                files = await self._fetch_file_tree(session, repo_id)
        else:
            files = await self._fetch_file_tree(session, repo_id)
            
        if not files:
            logger.debug(f"[ModelScope] No files found in {repo_id}")
            return []
        
        # Match Files
        for file_info in files:
            file_path = file_info.get("Path")
            file_type = file_info.get("Type")
            
            # Debug
            # print(f"[ModelScopeDebug] {repo_id} -> {file_path} ({file_type})")
            
            if not file_path or file_type != "blob": continue
            if not file_path.endswith(('.safetensors', '.ckpt', '.pt', '.pth', '.bin', '.gguf')): continue
            
            fname_base = os.path.splitext(os.path.basename(file_path))[0].lower()
            fname_ext = os.path.splitext(file_path)[1].lower()
            original_ext = os.path.splitext(original_filename)[1].lower()

            # Simple fuzzy match
            from rapidfuzz import fuzz
            p_score = fuzz.partial_ratio(original_base, fname_base)

            # [Bugfix] 扩展名一致性奖惩：请求 .safetensors 时优先返回同扩展名候选，
            # 避免原始仓库的 .pth 文件排在 Comfy-Org 的 .safetensors 前面。
            if original_ext and fname_ext:
                if fname_ext == original_ext:
                    p_score = min(100, int(p_score * 1.05))
                else:
                    p_score = int(p_score * 0.90)

            # print(f"[ModelScopeDebug] Checking {fname_base} vs {original_base} -> {p_score}")

            # If high confidence, add result
            if p_score > 60:
                # Direct Download Link Generation
                # Format: https://modelscope.cn/api/v1/models/{repo_id}/repo?Revision=master&FilePath={file_path}
                # ModelScope 默认分支可能是 master（Comfy-Org/unsloth 镜像常用），也可能是 main；
                # _fetch_file_tree 内部已做 master/main 回退，但下载链接固定用 master
                # 后续若发现 404 可再扩展分支动态选择。
                download_url = rewrite_modelscope_url(f"https://modelscope.cn/api/v1/models/{repo_id}/repo?Revision=master&FilePath={file_path}")

                results.append({
                    "source": "ModelScope (Direct)",
                    "name": f"{repo_id}/{os.path.basename(file_path)}",
                    "filename": os.path.basename(file_path),
                    "url": download_url,
                    "pageUrl": f"https://modelscope.cn/models/{repo_id}/files",
                    "score": p_score / 100.0
                })
        
        return results

    async def _fetch_file_tree(self, session, repo_id):
        """Call strict API /api/v1/models/.../repo/files"""
        url = f"{self.api_url}/models/{repo_id}/repo/files"
        
        # Try master then main
        revisions = ["master", "main"]
        
        for rev in revisions:
            params = {
                "Revision": rev,
                "Recursive": "True",
                "Root": ""
            }
            headers = {
                "Referer": f"https://modelscope.cn/models/{repo_id}/files"
            }
            
            try:
                resp = await session.get(url, params=params, headers=headers)
                if resp.status_code == 200:
                    data = resp.json()
                    files = data.get("Data", {}).get("Files", [])
                    
                    if files:
                        # Cache success
                        self._tree_cache[repo_id] = {"files": files, "ts": time.time()}
                        return files
            except Exception as e:
                logger.exception(f"[ModelScope] Tree fetch error {repo_id} ({rev}): {e}")
                
        return []

class LiblibProvider(BaseProvider):
    """
    Search models on liblib.art (哩布哩布) via its internal Web API.

    T2.4 重写说明（替换旧版失效的静态 HTML 抓取）：
    旧实现抓取 `liblib.art/search` 的静态 /modelinfo/ 链接，但搜索页是 JS 动态渲染，
    静态链接基本缺失 → 生产环境几乎失效。改为调用 liblib **内部 JSON API**
    （与前端同款，已被社区工具 liblib-spider 验证）：

      · 搜索：POST https://api2.liblib.art/api/www/model/search
            body: {"keyword","periodTime":["all"],"page":1,"pageSize":50,
                   "types":[],"models":[],"vipType":[]}
            resp: {"code":0, "data":{"data":[{uuid, name?, ...}], "hasMore":bool}}
            **只需浏览器 UA，无需登录态 / Token**（已验证匿名可调用）
      · 详情：POST https://api2.liblib.art/api/www/model/getByUuid/{uuid}
            resp: {"code":0, "data":{"name", "versions":[...]}}
            同样匿名可调用；本 Provider 只取 name 用于相似度打分，不下载。
      · 页面链接按 uuid 拼接：https://www.liblib.art/modelinfo/{uuid}

    维护提示：api2.liblib.art 为未公开内部接口，可能随站点改版变动；
    解析逻辑由 regression_tests/provider_check.py 的 mock 用例锁死，便于回归发现断裂。
    """
    def __init__(self, config):
        super().__init__(config)
        self.api_base = "https://api2.liblib.art/api/www"
        self.search_url = f"{self.api_base}/model/search"
        self.info_url_tpl = f"{self.api_base}/model/getByUuid/{{uuid}}"
        self.page_url_tpl = "https://www.liblib.art/modelinfo/{uuid}"
        # 内部接口只需浏览器 UA；curl_cffi 的 chrome124 指纹与站点前端一致即可。
        self.impersonate = "chrome124"

    async def _fetch_json(self, session, url, json_body=None):
        """GET/POST 取 JSON；非 200 或解析失败返回 None（调用方自行兜底）。"""
        params = {"timestamp": time.time()}
        headers = self._get_headers("https://www.liblib.art/")
        try:
            if json_body is not None:
                resp = await session.post(url, params=params, json=json_body, headers=headers)
            else:
                resp = await session.get(url, params=params, headers=headers)
        except Exception as e:
            logger.warning(f"[LiblibProvider] request failed: {e}")
            return None
        if resp.status_code != 200:
            logger.warning(f"[LiblibProvider] Status {resp.status_code} for {url}")
            return None
        try:
            return resp.json()
        except Exception:
            return None

    async def _get_model_name(self, session, model_uuid, fallback=None):
        """从详情接口取模型名；失败回退 fallback。"""
        info = await self._fetch_json(session, self.info_url_tpl.format(uuid=model_uuid))
        if info and info.get("code") == 0 and info.get("data"):
            return info["data"].get("name") or fallback
        return fallback

    async def search(self, query, original_filename):
        results = []
        try:
            logger.debug(f"[LiblibProvider] Searching: {query}")
            body = {
                "keyword": query,
                "periodTime": ["all"],
                "page": 1,
                "pageSize": 50,
                "types": [],
                "models": [],
                "vipType": [],
            }
            headers = self._get_headers("https://www.liblib.art/")
            async with AsyncSession(impersonate=self.impersonate, headers=headers, timeout=self.timeout) as session:
                search_resp = await self._fetch_json(session, self.search_url, json_body=body)
                if not search_resp:
                    return []
                items = (search_resp.get("data") or {}).get("data") or []
                original_lower = original_filename.lower()
                seen = set()
                candidates = []  # [(uuid, name)]
                for item in items[:10]:
                    model_uuid = item.get("uuid")
                    if not model_uuid or model_uuid in seen:
                        continue
                    seen.add(model_uuid)
                    candidates.append((model_uuid, item.get("name")))

                # 对缺失 name 的候选批量补名（最多 5 个，避免过多请求）
                missing = [(u, n) for (u, n) in candidates if not n]
                if missing:
                    top_missing = missing[:5]
                    names = await asyncio.gather(
                        *(self._get_model_name(session, u) for (u, _) in top_missing)
                    )
                    name_map = {u: nm for (u, _), nm in zip(top_missing, names)}
                    candidates = [(u, n or name_map.get(u) or u) for (u, n) in candidates]

                for model_uuid, name in candidates:
                    name_lower = (name or model_uuid).lower()
                    score = AdvancedTokenizer.calculate_similarity(original_lower, name_lower)
                    if score > 0.3:
                        page_url = self.page_url_tpl.format(uuid=model_uuid)
                        results.append({
                            "source": "Liblib",
                            "name": name or model_uuid,
                            "filename": "Direct Link (Click to Visit)",
                            "url": page_url,
                            "pageUrl": page_url,
                            "score": score,
                        })
        except Exception as e:
            logger.exception(f"[LiblibProvider] Error: {e}")
        return results

class CNBProvider(BaseProvider):
    """
    [v3.5.1] CNB (cnb.cool) Provider - Scrapes repositories from CNB ai-models group.
    CNB is a high-speed Git platform in China.
    """
    def __init__(self, config):
        super().__init__(config)
        self.search_url = "https://cnb.cool/ai-models/-/repos"

    async def search(self, query, original_filename):
        headers = self._get_headers("https://cnb.cool")
        results = []
        
        
        # [v3.5.2] CNB uses 'name' for repository search
        params = {"name": query}
        
        try:
            async with AsyncSession(impersonate="chrome124", headers=headers, timeout=15) as session:
                resp = await session.get(self.search_url, params=params)
                if resp.status_code == 200:
                    selector = Selector(text=resp.text)
                    # Repository links on CNB search page
                    # Format: <a href="/ai-models/username/repo">...</a>
                    repo_links = selector.xpath('//a[contains(@href, "/ai-models/")]/@href').getall()
                    
                    seen_repos = set()
                    for link in repo_links:
                        # Clean link: /ai-models/author/repo-name
                        # Link might be like: /ai-models/black-forest-labs/FLUX.1-schnell
                        parts = [p for p in link.split('/') if p and p != '-' and not p.startswith('?')]
                        # Parts: ['ai-models', 'author', 'repo']
                        if len(parts) >= 3 and parts[0] == 'ai-models':
                            author = parts[1]
                            repo = parts[2]
                            repo_id = f"{author}/{repo}"
                            if repo_id in seen_repos: continue
                            
                            # [v3.5.2] Relevance Check: CNB search is fuzzy, so verify keyword match
                            # Split query into terms (e.g. "wan 2.1" -> ["wan", "2.1"])
                            query_terms = [t.lower() for t in query.split('_') if len(t) > 1]
                            if not query_terms: query_terms = [query.lower()]
                            
                            repo_lower = repo_id.lower()
                            # Check if at least one major term is in the repo name
                            if not any(term in repo_lower for term in query_terms):
                                continue

                            seen_repos.add(repo_id)
                            
                            results.append({
                                "source": "CNB",
                                "name": repo_id,
                                "filename": original_filename,
                                "url": f"https://cnb.cool/ai-models/{repo_id}",
                                "pageUrl": f"https://cnb.cool/ai-models/{repo_id}",
                                "score": 0.85 
                            })
                            if len(results) >= 5: break
                            
        except Exception as e:
            logger.exception(f"[CNBProvider] Error: {e}")
            
        return results

class ModelSearcher:
    def __init__(self):
        self.config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "envs", "config.json")
        self.config = self.load_config()
        self.search_cache = {}
        
        # Provider 优先级 [v3.0.1]:
        # 1. CivitaiHashProvider (100% 精确匹配，需要本地文件路径)
        # 2. HuggingFace File Search (精确文件名匹配)
        # 3. Civitai (文本搜索) > HuggingFace API > Liblib > ModelScope > CNB (结构化源兜底)
        #    [P6] 已移除 Google / DuckDuckGo 泛网页搜索分支（重、易碎、无结构化结果）

        self.hash_provider = CivitaiHashProvider(self.config)  # [v3.0.1] SHA256 精确匹配

        self.providers = [
            HuggingFaceFileSearchProvider(self.config),  # [v3.0] 精确文件名搜索
            CivitaiProvider(self.config),
            HuggingFaceProvider(self.config),
            LiblibProvider(self.config),
            ModelScopeFileSearchProvider(self.config),   # [v3.5.0] ModelScope Direct File Search
            CNBProvider(self.config), # [v3.5.2] CNB Provider
        ]

        # [Bugfix] 组件类别词正则：名称含这些词的候选属于「组件」（text_encoder /
        # vae / clip / dav 等），不应作为主模型（diffusion_models / unet / checkpoints）的匹配结果返回。
        # 注意：正则 ``\b`` 把 ``_`` 视为单词字符，而文件名组件词常以 ``_`` 连接
        # （如 ``text_encoder`` / ``music3_dav``），``\b`` 锚点在此处失效。故改用
        # 显式边界：组件词前后必须是字符串边界或非字母数字分隔符（``_-.`` 等）。
        self._component_re = re.compile(
            r"(?i)(?<![a-z0-9])("
            r"text[ _.-]?encoder|textencoders|vae|clip|encoder|"
            r"tokenizer|embed(dings)?|dav|lm[ _.-]?head|adaptor"
            r")(?![a-z0-9])"
        )

    # ----------------------------------------------------------
    # [Bugfix] 组件类别过滤 + 源优先级加权
    # ----------------------------------------------------------
    def _is_component_name(self, name):
        """判断文件名是否指向模型组件（text_encoder / vae / clip / dav 等）。"""
        base = os.path.splitext(os.path.basename(name or ""))[0]
        return bool(self._component_re.search(base))

    def _filter_component_candidates(self, filename, candidates):
        """主模型请求不应返回 text_encoder / vae / clip / dav 等组件文件。

        - 若目标文件名本身含组件关键词（如 ``..._vae`` / ``..._text_encoder``），
          视为组件请求，保留组件候选（用户就是要这个组件）。
        - 否则为主模型请求，剔除所有组件候选，避免主模型被同名组件的官方镜像
          文件（如 ``minimax_music3_text_encoder_pruned_int8_convrot``、
          ``minimax_music3_dav``）抢答。
        """
        if self._is_component_name(filename):
            return candidates
        filtered = []
        for c in candidates:
            cand_name = c.get("filename") or c.get("name") or ""
            if self._is_component_name(cand_name):
                logger.debug("[AutoMatch] 丢弃组件候选（主模型请求）: %s", cand_name)
                continue
            filtered.append(c)
        return filtered

    def _apply_source_preference(self, candidates):
        """主模型优先 ComfyUI 官方国内镜像（HuggingFace / ModelScope 等）。

        通过 source 命中加权，让相近分数的候选中官方镜像源胜出；不改变绝对高分项。

        配置来源：优先读自身 ``self.config``（envs/config.json），
        缺失时回落到全局合并配置 ``get_all_config()``（core/data/matcher_config.json
        中的 ``searcher.source_preference``），保证在线源优先级在生产环境真正生效。
        """
        pref = self.config.get("searcher", {}).get("source_preference", {})
        if not pref:
            try:
                from .config import get_all_config
                pref = get_all_config().get("searcher", {}).get("source_preference", {})
            except Exception:
                pref = {}
        if not pref:
            return candidates
        for c in candidates:
            src = c.get("source", "") or ""
            weight = 1.0
            for key, w in pref.items():
                if key.lower() in src.lower():
                    weight = float(w)
                    break
            c["score"] = float(c.get("score", 0.0)) * weight
        return candidates

    def _apply_namespace_preference(self, candidates):
        """[Bugfix] 对 Comfy-Org / unsloth 等高优先级命名空间加权。

        即使 source 同为 HuggingFace 或 ModelScope，官方镜像组织的仓库也应优先于
        原始作者仓库或第三方非官方仓库。配置来源同 ``_apply_source_preference``。
        """
        pref = self.config.get("searcher", {}).get("namespace_preference", {})
        if not pref:
            try:
                from .config import get_all_config
                pref = get_all_config().get("searcher", {}).get("namespace_preference", {})
            except Exception:
                pref = {}
        if not pref:
            pref = {"Comfy-Org": 1.15, "unsloth": 1.08}
        # 将 priority_organizations 中的组织也纳入加权（默认 1.12），保证配置即生效，
        # 无需在 namespace_preference 中重复列举。
        try:
            from .config import get_all_config
            po = self.config.get("searcher", {}).get("priority_organizations", {})
            if not po:
                po = get_all_config().get("searcher", {}).get("priority_organizations", {})
            all_orgs = set()
            for orgs in (po or {}).values():
                all_orgs.update(orgs or [])
            for org in all_orgs:
                pref.setdefault(org, 1.12)
        except Exception:
            pass
        for c in candidates:
            name = c.get("name", "") or ""
            for ns, w in pref.items():
                if name.lower().startswith(ns.lower() + "/"):
                    c["score"] = float(c.get("score", 0.0)) * float(w)
                    break
        return candidates

    def _main_model_provider_order(self):
        """主模型默认 Provider 优先级：ComfyUI 官方国内镜像（HF / ModelScope / CNB）优先，
        其次 Civitai / Liblib 等社区源。其余（如未来扩展源）排末位。"""
        priority = ['huggingfacefilesearch', 'huggingface', 'modelscope', 'cnb']
        secondary = ['civitai', 'liblib']

        def sort_key(p):
            name = type(p).__name__.lower()
            for i, key in enumerate(priority):
                if key in name:
                    return (0, i)
            for i, key in enumerate(secondary):
                if key in name:
                    return (1, i)
            return (2, 0)

        return sorted(self.providers, key=sort_key)


    def load_config(self):
        if os.path.exists(self.config_path):
            try:
                with open(self.config_path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except: pass
        return {"civitai_api_key": ""}

    def get_config(self):
        return self.config

    def save_config(self, new_config):
        self.config.update(new_config)
        try:
            os.makedirs(os.path.dirname(self.config_path), exist_ok=True)
            with open(self.config_path, "w", encoding="utf-8") as f:
                json.dump(self.config, f, indent=4)
        except: pass
        
    async def validate_api_key(self, api_key):
        if not api_key: return False, "Empty API Key"
        # Validate using curl_cffi
        try:
            async with AsyncSession(impersonate="chrome120", timeout=10) as session:
                resp = await session.get("https://civitai.com/api/v1/models?limit=1", 
                                       headers={"Authorization": f"Bearer {api_key}"})
                if resp.status_code == 200: return True, "Valid API Key"
                if resp.status_code == 401: return False, "Invalid API Key"
                return False, f"Status: {resp.status_code}"
        except Exception as e:
            return False, str(e)

    async def search(self, filename, ignore_cache=False, file_path=None):
        """
        在线搜索模型匹配
        
        Args:
            filename: 原始文件名
            ignore_cache: 是否忽略缓存
            file_path: [v3.0.1] 可选，本地文件完整路径 (用于 SHA256 哈希匹配)
        """
        if not filename: return None

        # [Strict Filter] Verify extension
        from .scanner import is_valid_model_file
        if not is_valid_model_file(filename):
            logger.debug(f"[AutoMatch] Skipped non-model file: {filename}")
            return None
        
        if not ignore_cache and filename in self.search_cache:
            logger.debug(f"[AutoMatch] Cache Hit: {filename}")
            return self.search_cache[filename]
        
        # [v3.0.1] 优先尝试 SHA256 哈希匹配 (100% 精确)
        if file_path and os.path.exists(file_path):
            try:
                hash_results = await self.hash_provider.search_by_hash(file_path, filename)
                if hash_results:
                    best = hash_results[0]
                    logger.info(f"[AutoMatch] ✓ Hash Match: {best['name']} (100% accurate)")
                    self.search_cache[filename] = best
                    return best
            except Exception as e:
                logger.exception(f"[AutoMatch] Hash search failed: {e}")

        # [v3.5.1] Step 1: Normalize filename via Local DB & Popular Aliases
        # Instead of returning immediately, we use this for cross-provider resolution.
        normalized_filename = filename
        db_fallback_result = None
        
        # 1a. Popular Model Lookup (e.g. Wan2.1, Flux)
        repo_id, matched_key = AdvancedTokenizer.lookup_popular_model(filename)
        if repo_id:
            normalized_filename = matched_key if matched_key.endswith(('.safetensors', '.gguf')) else filename
            db_fallback_result = {
                "url": f"https://huggingface.co/{repo_id}/tree/main",
                "source": "HuggingFace (Official)",
                "name": repo_id,
                "pageUrl": f"https://huggingface.co/{repo_id}",
                "score": 1.0
            }

        # 1b. Database Match (if no popular hit or as refinement)
        if not db_fallback_result:
            db_match, score = find_best_match_in_db(filename)
            if db_match and score >= 0.85:
                normalized_filename = db_match["filename"]
                db_fallback_result = {
                    "source": f"HuggingFace ({db_match['source']} DB)",
                    "name": db_match["repo_id"],
                    "filename": db_match["filename"],
                    "url": db_match["url"],
                    "pageUrl": db_match["pageUrl"],
                    "score": score
                }

        search_terms = AdvancedTokenizer.extract_search_terms(normalized_filename)
        # Add original filename terms as fallback if normalized is different
        if normalized_filename != filename:
            orig_terms = AdvancedTokenizer.extract_search_terms(filename)
            for t in orig_terms:
                if t not in search_terms: search_terms.append(t)
        
        base_name = os.path.splitext(os.path.basename(normalized_filename))[0]
        
        # [v3.3.2] 方案 D: Provider 智能路由
        # 根据文件名特征选择优先 Provider
        import re as re_module
        has_chinese = bool(re_module.search(r'[\u4e00-\u9fff]', base_name))
        is_flux_wan_qwen = bool(re_module.search(r'(flux|wan|qwen|ltx|z[-_]?image)', base_name, re_module.IGNORECASE))
        
        if has_chinese:
            # 中文模型 -> 优先 Liblib/ModelScope/CNB
            priority_providers = [
                p for p in self.providers 
                if any(name in type(p).__name__.lower() for name in ['liblib', 'modelscope', 'cnb'])
            ]
            secondary_providers = [p for p in self.providers if p not in priority_providers]
            ordered_providers = priority_providers + secondary_providers
            logger.debug(f"[AutoMatch] 中文模型 -> 优先 Liblib/ModelScope/CNB")
        elif bool(re_module.search(r'(wan|qwen)', base_name, re_module.IGNORECASE)):
            # [v3.5.0] Wan/Qwen (国产优选) -> 优先 ModelScope/CNB
            priority_providers = [
                p for p in self.providers 
                if any(name in type(p).__name__.lower() for name in ['modelscope', 'cnb'])
            ]
            secondary_providers = [p for p in self.providers if p not in priority_providers]
            ordered_providers = priority_providers + secondary_providers
            logger.debug(f"[AutoMatch] Wan/Qwen (国产) -> 优先 ModelScope/CNB")
        elif bool(re_module.search(r'(flux|ltx|z[-_]?image)', base_name, re_module.IGNORECASE)):
            # GRAVITY-NOTE: Flux 等国际模型 -> 优先 HuggingFace
            priority_providers = [
                p for p in self.providers 
                if 'huggingface' in type(p).__name__.lower()
            ]
            secondary_providers = [p for p in self.providers if p not in priority_providers]
            ordered_providers = priority_providers + secondary_providers
            logger.debug(f"[AutoMatch] FLUX/Global -> 优先 HuggingFace")
            logger.debug(f"[AutoMatch] FLUX/Wan/Qwen 系列 -> 优先 HuggingFace")
        else:
            # [Bugfix] 主模型默认优先 ComfyUI 官方国内镜像（HF / ModelScope / CNB）
            ordered_providers = self._main_model_provider_order()
        
        # 只使用最优搜索词，所有 Provider 同时启动
        best_term = search_terms[0] if search_terms else base_name
        logger.info(f"[AutoMatch] Searching: {filename} | Term: {best_term}")
        
        import time
        start_time = time.time()
        
        all_candidates = []
        
        # 启动所有 Provider 任务 (使用智能路由后的顺序)
        tasks = [provider.search(best_term, base_name) for provider in ordered_providers]
        
        # 使用 as_completed 实现早停
        for future in asyncio.as_completed(tasks):
            try:
                res = await future
                if res and isinstance(res, list):
                    all_candidates.extend(res)
                    # [Bugfix] 主模型请求即时剔除组件候选，避免其触发早停抢答
                    all_candidates = self._filter_component_candidates(base_name, all_candidates)

                    # 早停检查: 任何 Provider 返回高分匹配立即停止
                    all_candidates.sort(key=lambda x: x.get("score", 0), reverse=True)
                    if all_candidates and all_candidates[0].get("score", 0) >= 0.7:
                        elapsed = time.time() - start_time
                        logger.info(f"[AutoMatch] Fast match in {elapsed:.2f}s: {all_candidates[0]['name']}")
                        break

            except Exception as e:
                logger.exception(f"[AutoMatch] Provider task failed: {e}")

        elapsed = time.time() - start_time
        logger.info(f"[AutoMatch] Completed in {elapsed:.2f}s")

        # [Bugfix] 组件类别二次过滤（幂等）+ 源/命名空间优先级加权
        all_candidates = self._filter_component_candidates(base_name, all_candidates)
        all_candidates = self._apply_source_preference(all_candidates)
        all_candidates = self._apply_namespace_preference(all_candidates)

        # Final Sort and Deduplication
        all_candidates.sort(key=lambda x: x.get("score", 0), reverse=True)
        
        unique_candidates = []
        seen_urls = set()
        for cand in all_candidates:
            url = cand.get("pageUrl", "")
            if url not in seen_urls:
                unique_candidates.append(cand)
                seen_urls.add(url)
                
        best_match = unique_candidates[0] if unique_candidates else None
        
        # [v3.5.1] If no high-score hit found online, check if we have a DB fallback
        if not best_match or best_match.get("score", 0) < 0.85:
            if db_fallback_result:
                logger.info(f"[AutoMatch] Returning DB fallback: {db_fallback_result['name']}")
                best_match = db_fallback_result

        if best_match:
            logger.info(f"[AutoMatch] Match Found: {best_match['name']} ({best_match['source']}) Score: {best_match.get('score', 0):.2f}")
            self.search_cache[filename] = best_match
            return [best_match]
        else:
            logger.info(f"[AutoMatch] No match for: {filename}")
            self.search_cache[filename] = None
            return []
