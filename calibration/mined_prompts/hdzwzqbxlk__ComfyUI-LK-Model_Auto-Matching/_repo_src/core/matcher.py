import difflib
import os
import re
try:
    from .config import get_matcher_config, get_features
except ImportError:
    from config import get_matcher_config, get_features

try:
    from .utils import AdvancedTokenizer
except ImportError:
    from utils import AdvancedTokenizer

class ModelMatcher:
    def __init__(self, scanner):
        self.scanner = scanner
        self.config = get_matcher_config()
        # 倒排索引: {token: set(model_indices)}
        self.inverted_index = {}
        self.model_list = [] # List storing actual model info, referenced by index
        self._index_built = False
        self._last_model_count = -1
        
        # Pre-compile regex for performance
        import re
        self.rank_pat = re.compile(r'(?:rank|step|epoch)[-_]?(\d+)')
        self.anchor_pat = re.compile(r"(?i)^(wan\d|sdxl|pony|flux)")
        self.ver_pat = re.compile(r"(?i)^(v\d|\d+\.\d+)")

        # [Bugfix] 本地匹配「核心词覆盖率」硬门槛配置
        # 候选必须覆盖目标核心身份词的显著比例，否则即便共享 fp16/safetensors
        # 等技术后缀也强制拒绝（解决 minimax…dit 误配 sam…multiplex 类问题）。
        matching_cfg_init = self.config.get('matching', {})
        self.core_coverage_min = float(matching_cfg_init.get('core_coverage_min', 0.6))
        self.core_min_tokens = int(matching_cfg_init.get('core_min_tokens', 2))

    def invalidate_index(self):
        """显式使索引失效，强制在下次匹配时重新构建"""
        self._index_built = False
        self._last_model_count = -1

    def _normalize_name(self, name):
        """标准化模型名称，移除扩展名并转小写"""
        base, _ = os.path.splitext(name)
        return base.lower().strip()

    def _get_basename(self, path):
        """提取纯文件名 (不含目录，不含扩展名)"""
        # handle both / and \ just in case
        name = os.path.basename(path.replace("\\", "/"))
        base, _ = os.path.splitext(name)
        return base.lower().strip()

    # ----------------------------------------------------------
    # [Bugfix] 本地匹配核心词覆盖率硬门槛
    # 目标文件名含明确核心身份词（如 minimax / music3 / dit / sam / multiplex）
    # 时，候选必须覆盖其中显著比例，技术后缀（fp16/fp8/bf16/safetensors…）不计入。
    # ----------------------------------------------------------
    _CORE_GENERIC = {'model', 'models', 'net', 'network', 'test',
                     'v', 'version', 'ver', 'm'}
    _CORE_NUM_RE = re.compile(r'^\d+(\.\d+)?$')

    def _core_identity_tokens(self, base_name):
        """提取文件名的核心身份词（排除技术后缀 / 纯数字 / 版本号 / 通用词）。

        用于「核心词覆盖率」硬门槛：候选必须覆盖目标核心身份词的显著比例，
        否则即便共享 fp16 / safetensors 等后缀也强制拒绝。

        [Latent-fix] 分词前先调用 ``_strip_variant_terms`` 剥离精度 / 量化 / 格式等
        技术变体词。原因：``AdvancedTokenizer.tokenize`` 会把 ``fp16`` / ``bf16``
        拆成 ``fp`` / ``bf`` + 数字（字母与数字分离），若直接对原串分词，
        ``fp`` / ``bf`` / ``f`` 这类片段会泄漏为「核心身份词」，导致单字基本体
        + 精度后缀（如 ``mycoolmodel_fp16``）被覆盖率门槛误杀。复用于变体匹配
        路径一致的 ``_strip_variant_terms``，从源头消除精度片段，并使覆盖率门槛
        与 ``get_core_tokens`` 的语义保持一致。
        """
        from .utils import AdvancedTokenizer, NOISE_SUFFIXES
        cleaned = AdvancedTokenizer._strip_variant_terms(base_name)
        tokens = set(AdvancedTokenizer.tokenize(cleaned))
        core = set()
        for t in tokens:
            if t in NOISE_SUFFIXES:
                continue
            if t in self._CORE_GENERIC:
                continue
            if t.isdigit():
                continue
            if self._CORE_NUM_RE.match(t):
                continue
            if len(t) < 2:
                continue
            core.add(t)
        return core

    def _core_coverage_ok(self, target_base, cand_base):
        """核心词覆盖率门槛：候选必须覆盖目标核心身份词的 ``core_coverage_min`` 比例。

        仅当目标本身含有足够明确的身份词（>= ``core_min_tokens``）时才强制门槛；
        身份词不足（如极简文件名）时交由既有评分逻辑判定，避免误杀正常变体匹配
        （如 ``flux1-dev`` 与 ``flux1-dev-fp8``）。
        """
        target_core = self._core_identity_tokens(target_base)
        if len(target_core) < self.core_min_tokens:
            return True
        cand_core = self._core_identity_tokens(cand_base)
        if not cand_core:
            return False
        covered = len(target_core & cand_core)
        return (covered / len(target_core)) >= self.core_coverage_min

    def _build_index(self, force=False):
        """构建倒排索引以加速匹配 (O(N) -> O(1))，具备缓存校验功能"""
        current_models = list(self.scanner.get_all_models())
        current_count = len(current_models)
        
        # 如果索引已建立且模型数量一致，且未强制刷选，则复用现有倒排索引
        if self._index_built and not force and current_count == self._last_model_count:
            return

        self.model_list = current_models
        self._last_model_count = current_count
        self.inverted_index = {}
        
        # [v3.2.0] Phase 3: 格式分区索引
        self.format_indices = {
            "gguf": set(),
            "standard": set(),
            "other": set()
        }
        
        for idx, info in enumerate(self.model_list):
            filename = info["filename"]
            
            # 1. 格式分区
            ext = os.path.splitext(filename)[1].lower()
            if ext == ".gguf":
                self.format_indices["gguf"].add(idx)
            elif ext in {".safetensors", ".ckpt", ".pt", ".pth", ".bin"}:
                self.format_indices["standard"].add(idx)
            else:
                self.format_indices["other"].add(idx)
            
            # 2. Token 倒排索引
            base_tokens = AdvancedTokenizer.tokenize(self._get_basename(filename))
            
            for token in base_tokens:
                if token not in self.inverted_index:
                    self.inverted_index[token] = set()
                self.inverted_index[token].add(idx)
                
        self._index_built = True

    def match(self, missing_items):
        """
        匹配缺失的模型 (Refactored for Clean Code)
        """
        matches = []
        self._build_index()
        
        # Build maps one time
        full_name_map = {}
        basename_map = {}
        for info in self.model_list:
            filename = info["filename"]
            full_name_map[self._normalize_name(filename)] = info
            full_name_map[filename.lower()] = info
            base = self._get_basename(filename)
            if base not in basename_map:
                basename_map[base] = info
        
        # Prepare Context
        ctx = {
            "full_name_map": full_name_map,
            "basename_map": basename_map,
            "WIDGET_TO_TYPE": {
                "ckpt_name": ["checkpoints", "unet", "diffusion_models"],
                "unet_name": ["unet", "diffusion_models", "checkpoints"],
                "model_name": ["unet", "diffusion_models", "checkpoints"],
                "diffusion_model": ["diffusion_models", "unet", "checkpoints"],
                "lora_name": ["loras"],
                "vae_name": ["vae"],
                "clip_name": ["clip", "text_encoders"],
                "text_encoder_name": ["text_encoders", "clip"],
                "control_net_name": ["controlnet", "t2i_adapter"],
                "controlnet_name": ["controlnet", "t2i_adapter"],
                "upscale_model_name": ["upscale_models"],
                "embeddings_name": ["embeddings"],
                "embedding_name": ["embeddings"],
                "style_model_name": ["style_models"],
                "hypernetwork_name": ["hypernetworks"],
                "gligen_name": ["gligen"],
            }
        }

        for item in missing_items:
            current_val = item.get("current")
            if not current_val: continue
            
            # [Filter] Skip non-model files
            from .scanner import is_valid_model_file
            if not is_valid_model_file(current_val): continue
            
            # Prepare Item Context
            item_ctx = {
                "current_val": current_val,
                "target_norm": self._normalize_name(current_val),
                "target_base": self._get_basename(current_val),
                "widget_name": item.get("widget_name", ""),
                "expected_types": ctx["WIDGET_TO_TYPE"].get(item.get("widget_name", ""), [])
            }
            
            best_match = None
            match_type = "Unknown"

            matching_cfg = self.config.get('matching', {})
            db_cfg = self.config.get('db', {})

            # 1. Exact Match (本地索引优先：用户本地模型第一顺位)
            if not best_match and matching_cfg.get('use_exact_match', True):
                exact = self._find_exact_match(item_ctx, ctx)
                if exact:
                    best_match = exact
                    match_type = "Exact"
             
            # 2. Fuzzy Match
            if not best_match and matching_cfg.get('use_fuzzy_match', True):
                fuzzy = self._find_fuzzy_match(item_ctx)
                if fuzzy:
                    best_match = fuzzy
                    match_type = "Fuzzy"
             
            # 3. Variant Match
            if not best_match and matching_cfg.get('use_variant_match', True):
                variant = self._find_variant_match(item_ctx)
                if variant:
                    best_match = variant
                    match_type = "Variant"
             
            # 4. Legacy Match
            if not best_match and matching_cfg.get('use_legacy_match', True):
                legacy = self._find_legacy_match(item_ctx, ctx)
                if legacy:
                    best_match = legacy
                    match_type = "Fuzzy" # Legacy is technically fuzzy

            # 5. Network/DB fallback（T2.5 本地优先）：本地四层完全无匹配时才启用。
            #    此块必须为最后一道，避免本地已有精确文件却被外部标准名抢答。
            #    [Bugfix] 配置键由 use_db_first 更名为 use_db_fallback，语义明确为
            #    「本地完全无匹配才回退 DB（ComfyUI 官方镜像信息）」，杜绝命名歧义。
            if not best_match and matching_cfg.get('use_db_fallback', True):
                try:
                    from .database import db
                    db_match, db_score = db.lookup_modelsdb(
                        current_val,
                        expected_types=item_ctx['expected_types']
                    )
                    if db_match and db_score >= db_cfg.get('semantic_min_score', 0.35):
                        # quick conflict/type check
                        if not self._check_conflicts(current_val, db_match.get('filename', '')):
                            cand_type = db_match.get('type', 'unknown')
                            expected_types = item_ctx['expected_types']
                            if not expected_types or cand_type in expected_types or cand_type == 'unknown':
                                best_match = db_match
                                match_type = "DB"
                except Exception:
                    # DB not available or error -> give up
                    pass

            if best_match:
                # ensure path/filename keys exist
                matched_filename = best_match.get("filename") or os.path.basename(best_match.get("path", ""))
                matched_path = best_match.get("path", matched_filename)

                matches.append({
                    "id": item["id"],
                    "node_type": item["node_type"],
                    "widget_name": item["widget_name"],
                    "original_value": current_val,
                    "matched_value": os.path.normpath(matched_filename),
                    "path": os.path.normpath(matched_path),
                    "match_type": match_type,
                    "type": best_match.get("type", "unknown")
                })
        
        return matches

    def _find_exact_match(self, item_ctx, ctx):
        """Priority 1 & 2: Exact Full Path or Basename Match"""
        target_norm = item_ctx["target_norm"]
        current_val = item_ctx["current_val"]
        target_base = item_ctx["target_base"]
        full_name_map = ctx["full_name_map"]
        basename_map = ctx["basename_map"]

        if target_norm in full_name_map:
            return full_name_map[target_norm]
        elif current_val.lower() in full_name_map:
            return full_name_map[current_val.lower()]
        elif target_base in basename_map:
            return basename_map[target_base]
        return None

    def _check_conflicts(self, target_name, candidate_name):
        """
        Check for hard conflicts. Returns True if conflict exists.
        Logic: Mutual Inclusion for Critical Tokens.
        If Token A is in Target, it MUST be in Candidate (and vice versa) for the match to be valid.
        """
        t_lower = target_name.lower()
        c_lower = candidate_name.lower()
        
        # [v3.2.0] Phase 1: HARD FORMAT BLOCK
        # .gguf and .safetensors/.ckpt are fundamentally different ecosystems
        FORMAT_GROUPS = {
            "gguf": {".gguf"},
            "standard": {".safetensors", ".ckpt", ".pt", ".pth", ".bin"}
        }
        
        def get_format_group(name):
            ext = os.path.splitext(name)[1].lower()
            for group, exts in FORMAT_GROUPS.items():
                if ext in exts:
                    return group
            return "other"
        
        t_group = get_format_group(target_name)
        c_group = get_format_group(candidate_name)
        
        if t_group != "other" and c_group != "other" and t_group != c_group:
            return True  # HARD CONFLICT: gguf <-> standard

        # [T2.2] 版本/族谱感知冲突（gated by features.version_aware）
        if self._version_aware_enabled():
            if self._version_family_conflict(t_lower, c_lower):
                return True

        # 1. Critical Token Mutual Exclusion
        # If one has it, the other MUST have it.
        # Format: (TokenString, IsStrictWordBoundary)
        # For now, simple substring is effective for these specific keys
        critical_tokens = [
            "i2v", "t2v",
            "inpainting",
            "vae", # Critical: VAE vs Checkpoint
            "upscaler", "upscale",  # 基类 vs 放大模型必须互斥（修复 qwen_image_edit upscale↔base 误匹配）
            "refiner",
            "img2vid", "txt2vid"
        ]
        
        # Helper to check presence (could be regex for boundary if needed, but filenames vary)
        # Using simple 'in' is safer for "Wan2.1_I2V" (no spaces)
        
        for token in critical_tokens:
            in_target = self._token_present(t_lower, token)
            in_cand = self._token_present(c_lower, token)

            # XOR: One has it, the other doesn't -> Conflict
            if in_target != in_cand:
                return True
                
        # 2. Mutually Exclusive Pairs (One is A, One is B -> Conflict)
        # Useful for things that aren't binary "presence" but specific alternatives
        conflict_pairs = [
            ("sdxl", "sd1.5"),
            ("mp4", "gif"),
            ("fp16", "fp8"), # User might care, but maybe fuzzy match handles score?
                             # Let's be strict for accuracy request.
            ("rank128", "rank64"), # Explicit Ranks
            ("rank128", "rank32"),
            ("rank64", "rank32"),
            ("rank83", "rank128"), # Specific user case
        ]
        
        for a, b in conflict_pairs:
            has_a_t = a in t_lower
            has_b_t = b in t_lower
            has_a_c = a in c_lower
            has_b_c = b in c_lower
            
            # If Target is A and Candidate is B -> Conflict
            if has_a_t and has_b_c: return True
            # If Target is B and Candidate is A -> Conflict
            if has_b_t and has_a_c: return True
            
        # 3. Numeric Rank/Step Extraction (General Case)
        # Extract all numbers preceded by 'rank' or 'step'
        t_nums = set(self.rank_pat.findall(t_lower))
        c_nums = set(self.rank_pat.findall(c_lower))
        
        # If both have extracted numbers, and they are disjoint -> Conflict
        if t_nums and c_nums and t_nums.isdisjoint(c_nums):
            # e.g. Target={128}, Cand={64} -> Conflict
            # e.g. Target={128}, Cand={128} -> OK
            return True

        return False

    def _token_present(self, name_lower, token):
        """词边界感知的 token 检测：token 必须作为独立词出现。

        避免 ``0.9vae`` 这类质量后缀被误判为 VAE 模型类型（朴素 ``in`` 子串会命中），
        同时保证 ``sdxl_vae`` / ``qwen..._upscale`` 等真实类型词仍被正确识别。
        独立词 = token 前后为分隔符（非字母数字）或字符串边界。
        """
        if not token:
            return False
        n = len(token)
        idx = 0
        while True:
            i = name_lower.find(token, idx)
            if i == -1:
                return False
            left_ok = (i == 0) or (not name_lower[i - 1].isalnum())
            j = i + n
            right_ok = (j == len(name_lower)) or (not name_lower[j].isalnum())
            if left_ok and right_ok:
                return True
            idx = i + n

    def _version_aware_enabled(self):
        """读取 features.version_aware 开关（异常时保守关闭）。"""
        try:
            return bool(get_features().get('version_aware', False))
        except Exception:
            return False

    def _family_variant(self, lower):
        """返回 (family, variant) 用于族谱变体冲突判定。

        family: 'flux' | 'sdxl' | None
        variant: 'dev'|'schnell'|'fill'|'kontext'|'canny' | 'base'|'refiner'|'instruct' | None
        仅当 family 显式出现时才返回非 None（避免 pony/xl 误判为 sdxl 族）。
        """
        if 'flux' in lower:
            variant = None
            if 'dev' in lower:
                variant = 'dev'
            elif 'schnell' in lower:
                variant = 'schnell'
            elif 'fill' in lower:
                variant = 'fill'
            elif 'kontext' in lower:
                variant = 'kontext'
            elif 'canny' in lower:
                variant = 'canny'
            return ('flux', variant)
        if 'sdxl' in lower:
            variant = None
            if 'base' in lower:
                variant = 'base'
            elif 'refiner' in lower:
                variant = 'refiner'
            elif 'instruct' in lower:
                variant = 'instruct'
            return ('sdxl', variant)
        return (None, None)

    def _version_family_conflict(self, t_lower, c_lower):
        """[T2.2] 同族不同版本 / 同族不同变体 => 硬冲突。

        与 utils.calculate_similarity 的版本感知语义保持一致：
        - 族内主/次版本不同（wan2.1 vs wan2.2）视为冲突；
        - flux dev/schnell、sdxl base/refiner 仅在「两侧都明确指定且不同」时冲突
          （单侧未指定变体不冲突，兼容泛化匹配）。
        """
        from .utils import AdvancedTokenizer

        # 1. 族谱变体冲突（flux / sdxl）
        fam_t, variant_t = self._family_variant(t_lower)
        fam_c, variant_c = self._family_variant(c_lower)
        if fam_t and fam_t == fam_c:
            # 仅当两侧都明确指定变体且不一致时冲突
            if variant_t and variant_c and variant_t != variant_c:
                return True

        # 2. 版本冲突（同一 family，主/次版本不同）
        fam_t2, maj_t, min_t = AdvancedTokenizer.parse_version_tuple(t_lower)
        fam_c2, maj_c, min_c = AdvancedTokenizer.parse_version_tuple(c_lower)
        if fam_t2 and fam_c2 and fam_t2 == fam_c2:
            if maj_t is not None and maj_c is not None and maj_t != maj_c:
                return True
            if min_t is not None and min_c is not None and min_t != min_c:
                return True

        return False

    def _find_fuzzy_match(self, item_ctx):
        """Priority 3: Token-based Fuzzy Match with Weights"""
        current_val = item_ctx["current_val"]
        target_base = item_ctx["target_base"]
        expected_types = item_ctx["expected_types"]
        
        target_tokens = AdvancedTokenizer.tokenize(target_base)
        candidate_indices = set()
        for token in target_tokens:
            if token in self.inverted_index:
                candidate_indices.update(self.inverted_index[token])
        
        # [v3.2.0] Phase 3: 格式预过滤
        target_ext = os.path.splitext(current_val)[1].lower()
        if target_ext == ".gguf":
            format_pool = self.format_indices.get("gguf", set())
        elif target_ext in {".safetensors", ".ckpt", ".pt", ".pth", ".bin"}:
            format_pool = self.format_indices.get("standard", set())
        else:
            format_pool = None  # 不过滤
        
        if format_pool is not None:
            candidate_indices = candidate_indices.intersection(format_pool)
        
        if not candidate_indices:
            return None

        # Weights & Anchors
        import re
        W_ANCHOR = 10.0
        W_VERSION = 5.0
        W_NORMAL = 1.0
        # [Bugfix] 技术后缀权重进一步压低：fp16/fp8/safetensors 等不应单独支撑一次匹配
        W_NOISE = 0.05
        from .utils import NOISE_SUFFIXES
        
        target_anchors = {t for t in target_tokens if self.anchor_pat.match(t)}
        target_versions = {t for t in target_tokens if self.ver_pat.match(t)}
        
        target_fmt = AdvancedTokenizer.get_model_format(current_val)
        if target_fmt == "other":
            if "gguf" in target_base.lower(): target_fmt = "gguf"
            elif ".safetensors" in current_val or ".ckpt" in current_val: target_fmt = "checkpoint"

        # Calculate Max Score for Normalization
        max_possible_score = 0.0
        for token in target_tokens:
            if token in target_anchors: max_possible_score += W_ANCHOR
            elif token in target_versions: max_possible_score += W_VERSION
            elif token in NOISE_SUFFIXES: max_possible_score += W_NOISE
            else: max_possible_score += W_NORMAL
        
        if max_possible_score <= 0: return None

        best_score = 0.0
        best_candidate = None

        for idx in candidate_indices:
            info = self.model_list[idx]
            
            # [Optimization] Quick conflict check before expensive token calc
            if self._check_conflicts(current_val, info["filename"]):
                continue
            
            cand_base = self._get_basename(info["filename"])
            cand_tokens = set(AdvancedTokenizer.tokenize(cand_base))

            # [Bugfix] 核心词覆盖率硬门槛：技术后缀（fp16 等）不应单独支撑匹配
            if not self._core_coverage_ok(target_base, cand_base):
                continue

            # 1. Base Score
            score = 0.0
            for token in target_tokens:
                if token in cand_tokens:
                    if token in target_anchors: score += W_ANCHOR
                    elif token in target_versions: score += W_VERSION
                    elif token in NOISE_SUFFIXES: score += W_NOISE
                    else: score += W_NORMAL
            
            base_final = (score / max_possible_score) * 100
            
            # 2. [v3.2.0] Multiplicative Format Penalty (Phase 2)
            # Format mismatch = hard zero, not weak penalty
            cand_fmt = AdvancedTokenizer.get_model_format(info["filename"])
            format_multiplier = 1.0
            if target_fmt != "other" and cand_fmt != "other" and target_fmt != cand_fmt:
                format_multiplier = 0.0  # HARD ZERO
            
            # 3. [Fix] Strict Type Enforcement
            type_score = 0.0
            cand_type = info.get("type", "unknown")
            if expected_types:
                if cand_type not in expected_types:
                    continue  # Strict Skip
                # 轻量一致性加分：仅作为同类型候选间的微小偏好，
                # 不可单独把弱名称匹配（base_final 很低）抬过阈值——否则
                # totally_unknown_model 会因共享 'model' 词 + 同类型而被误匹配。
                type_score = 10.0

            # 4. [v3.6.0] CJK 中文字符重叠 Bonus
            cjk_bonus = 0.0
            t_cjk = set(re.findall(r'[\u4e00-\u9fff]', target_base))
            c_cjk = set(re.findall(r'[\u4e00-\u9fff]', cand_base))
            if t_cjk and c_cjk:
                common_cjk = t_cjk.intersection(c_cjk)
                if common_cjk:
                    cjk_bonus = (len(common_cjk) / min(len(t_cjk), len(c_cjk))) * 25.0
            
            final_score = (base_final + type_score + cjk_bonus) * format_multiplier
            
            if final_score > best_score:
                best_score = final_score
                best_candidate = info
        
        if best_score >= self.config.get('matching', {}).get('fuzzy_score_cutoff', 60.0):
            return best_candidate
        return None

    def _find_variant_match(self, item_ctx):
        """Priority 4: Variant Match (Core Tokens Jaccard)"""
        target_base = item_ctx["target_base"]
        current_val = item_ctx["current_val"]
        expected_types = item_ctx["expected_types"]
        
        target_core = AdvancedTokenizer.get_core_tokens(target_base)
        if not target_core: return None
        
        target_fmt = AdvancedTokenizer.get_model_format(current_val)
        
        variant_indices = set()
        for token in target_core:
            if token in self.inverted_index:
                variant_indices.update(self.inverted_index[token])
        
        if not variant_indices: return None

        best_score = 0.0
        best_candidate = None

        for idx in variant_indices:
            info = self.model_list[idx]
            
            # [Optimization] Conflict Check
            if self._check_conflicts(current_val, info["filename"]):
                continue
            
            # Category Check
            cand_type = info.get("type", "unknown")
            if expected_types and cand_type not in expected_types:
                continue
            
            # Format Check
            cand_fmt = AdvancedTokenizer.get_model_format(info["filename"])
            if target_fmt != "other" and cand_fmt != "other" and target_fmt != cand_fmt:
                continue
                
            cand_base = self._get_basename(info["filename"])
            cand_core = AdvancedTokenizer.get_core_tokens(cand_base)
            if not cand_core: continue

            # [Bugfix] 核心词覆盖率硬门槛：技术后缀（fp16 等）不应单独支撑匹配
            if not self._core_coverage_ok(target_base, cand_base):
                continue

            intersection = len(target_core.intersection(cand_core))
            union = len(target_core.union(cand_core))
            core_score = intersection / union if union > 0 else 0.0
            
            if core_score > best_score:
                best_score = core_score
                best_candidate = info
        
        if best_score >= self.config.get('matching', {}).get('variant_score_cutoff', 0.9):
            return best_candidate
        return None

    def _find_legacy_match(self, item_ctx, ctx):
        """Priority 5: RapidFuzz Backup (Replaces slow difflib)"""
        try:
            from rapidfuzz import process, fuzz
        except ImportError:
            # Fallback if rapidfuzz missing (though it's in requirements)
            import difflib
            return self._find_legacy_match_difflib(item_ctx, ctx)

        target_base = item_ctx["target_base"]
        current_val = item_ctx["current_val"]
        expected_types = item_ctx["expected_types"]
        basename_map = ctx["basename_map"]
        
        available_names = list(basename_map.keys())
        
        # [v3.6.0] 提升中文与复杂前后缀模型比对支持
        # 使用 token_set_ratio，容忍带有作者前缀、目录层级、版本描述的中英文混排
        match = process.extractOne(
            target_base, 
            available_names, 
            scorer=fuzz.token_set_ratio, 
            score_cutoff=self.config.get('matching', {}).get('legacy_score_cutoff', 75)
        )
        
        if match:
            best_name, score, idx = match
            candidate_info = basename_map[best_name]

            # [Bugfix] 核心词覆盖率硬门槛：不能仅凭后缀相同（token_set_ratio 高）就匹配
            if not self._core_coverage_ok(target_base, best_name):
                return None

            # [Fix] Apply Conflict Check
            if self._check_conflicts(current_val, candidate_info["filename"]):
                return None
            
            # Category Check
            cand_type = candidate_info.get("type", "unknown")
            if expected_types and cand_type not in expected_types:
                return None
                
            return candidate_info
            
        return None

    def _find_legacy_match_difflib(self, item_ctx, ctx):
        """Fallback for when rapidfuzz is missing"""
        target_base = item_ctx["target_base"]
        current_val = item_ctx["current_val"]
        expected_types = item_ctx["expected_types"]
        basename_map = ctx["basename_map"]
        
        available_names = list(basename_map.keys())
        similars = difflib.get_close_matches(target_base, available_names, n=1, cutoff=0.85)
        
        if similars:
            match_info = basename_map[similars[0]]
            
            # [Fix] Apply Conflict Check here too
            if self._check_conflicts(current_val, match_info["filename"]):
                return None
                
            cand_type = match_info.get("type", "unknown")
            if expected_types and cand_type not in expected_types:
                return None
            return match_info
        return None
