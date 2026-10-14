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
    
    def __init__(self, config):
        super().__init__(config)
        self.api_url = "https://modelscope.cn/api/v1"
        self.timeout = 10
        self.impersonate = "chrome124"

    def _get_headers(self, referer=None):
        headers = super()._get_headers(referer)
        # ModelScope specific headers
        headers["Origin"] = "https://modelscope.cn"
        if referer:
            headers["Referer"] = referer
        return headers

    async def search(self, query, original_filename):
        """
        Implementation Strategy:
        1. Search for repositories using the general search API (similar to existing ModelScopeProvider)
        2. For top candidates, fetch file lists using the RE API
        3. Match files against original_filename
        """
        import time
        import uuid
        
        results = []
        original_lower = original_filename.lower()
        original_base = os.path.splitext(original_filename)[0].lower()
        
        # 1. Search for Repositories
        search_url = f"{self.api_url}/dolphin/models"
        search_payload = {
            "PageSize": 10,
            "PageNumber": 1,
            "SearchText": query,
            "Sort": {"SortBy": "Default"}
        }
        
        try:
            print(f"[ModelScope] Searching Repos for: {query}")
            headers = self._get_headers(referer="https://modelscope.cn/models")
            headers["Content-Type"] = "application/json"
            
            async with AsyncSession(impersonate=self.impersonate, headers=headers, timeout=self.timeout) as session:
                resp = await session.put(search_url, json=search_payload)
                if resp.status_code != 200:
                    print(f"[ModelScope] Search API Error: {resp.status_code}")
                    return []
                
                data = resp.json()
                if not data.get("Success"): return []
                
                models = data.get("Data", {}).get("Model", {}).get("Models", [])
                
                # 2. Iterate Repos and Fetch Files
                tasks = []
                for model in models:
                    repo_id = model.get("Path") # e.g. "AI-ModelScope/Wan-Video"
                    if not repo_id: continue
                    
                    # Quick filter on repo name relevance?
                    # For now, search top 5 results deeply
                    tasks.append(self._scan_repo_files(session, repo_id, original_base))
                    if len(tasks) >= 5: break
                
                if tasks:
                    start_time = time.time()
                    repo_results = await asyncio.gather(*tasks)
                    elapsed = time.time() - start_time
                    print(f"[ModelScope] Scanned {len(tasks)} repos in {elapsed:.2f}s")
                    
                    for res_list in repo_results:
                        if res_list:
                            results.extend(res_list)
                            
        except Exception as e:
            print(f"[ModelScope] Error: {e}")
            
        return sorted(results, key=lambda x: x.get("score", 0), reverse=True)[:5]

    async def _scan_repo_files(self, session, repo_id, original_base):
        """Fetch files for a specific repo using hidden API"""
        import time
        results = []
        
        # Check Cache
        now = time.time()
        if repo_id in self._tree_cache:
            cache = self._tree_cache[repo_id]
            if now - cache["ts"] < self.CACHE_TTL:
                files = cache["files"]
            else:
                files = await self._fetch_file_tree(session, repo_id)
        else:
            files = await self._fetch_file_tree(session, repo_id)
            
        if not files: return []
        
        # Match Files
        for file_info in files:
            file_path = file_info.get("Path") # e.g. "wan2.1_i2v.safetensors"
            if not file_path: continue
            
            # Use shared matching logic if possible? For now simple fuzzy
            fname_base = os.path.splitext(os.path.basename(file_path))[0].lower()
            
            # Simple fuzzy match for now (importing AdvancedTokenizer might be circular if not careful)
            # But we can use rudimentary check
            from utils import AdvancedTokenizer
            score = AdvancedTokenizer.calculate_similarity(original_base, fname_base)
            
            if score > 0.4:
                # Direct Download Link Generation
                # Format: https://modelscope.cn/api/v1/models/{repo_id}/repo?Revision=master&FilePath={file_path}
                # Note: This usually triggers a 302 redirect to the actual CDN
                download_url = f"https://modelscope.cn/api/v1/models/{repo_id}/repo?Revision=master&FilePath={file_path}"
                
                results.append({
                    "source": "ModelScope (Direct)",
                    "name": f"{repo_id} - {file_path}",
                    "filename": os.path.basename(file_path),
                    "url": download_url,
                    "pageUrl": f"https://modelscope.cn/models/{repo_id}/files",
                    "score": score
                })
        
        return results

    async def _fetch_file_tree(self, session, repo_id):
        """Call strict API /api/v1/models/.../repo/files"""
        url = f"{self.api_url}/models/{repo_id}/repo/files"
        params = {
            "Revision": "master", # Default to master? Or retrieve default branch?
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
                
                # Cache success
                self._tree_cache[repo_id] = {"files": files, "ts": time.time()}
                return files
            else:
                return []
        except Exception as e:
            print(f"[ModelScope] Tree fetch error {repo_id}: {e}")
            return []
