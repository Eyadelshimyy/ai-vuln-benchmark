import os
import json

try:
    from .mirror import rewrite_hf_url
except ImportError:
    from mirror import rewrite_hf_url

DATA_FILE = os.path.join(os.path.dirname(__file__), 'data', 'models_db.json')

_models_db = None
_civitai_map = None
_all_keys = None


def _load():
    global _models_db, _civitai_map, _all_keys
    if _models_db is not None:
        return
    if not os.path.exists(DATA_FILE):
        _models_db = {}
        _civitai_map = {}
        _all_keys = []
        return
    try:
        with open(DATA_FILE, 'r', encoding='utf-8') as f:
            payload = json.load(f)
        _models_db = payload.get('MODELS_DB', {})
        _civitai_map = payload.get('CIVITAI_MAP', {})
        _all_keys = list(_models_db.keys())
    except Exception:
        _models_db = {}
        _civitai_map = {}
        _all_keys = []


def _enrich_info(info):
    new_info = info.copy()
    repo_id = info.get('repo_id')
    path = info.get('path')
    if repo_id and path:
        new_info['url'] = rewrite_hf_url(f"https://huggingface.co/{repo_id}/resolve/main/{path}")
        new_info['pageUrl'] = f"https://huggingface.co/{repo_id}/tree/main"
    return new_info


def find_best_match_in_db(filename: str):
    """Lookup best match from JSON models DB.
    Returns (matched_info_dict, score) or (None, 0)
    """
    _load()
    if not filename:
        return (None, 0)

    base = os.path.basename(filename)
    base_lower = base.lower()
    name_no_ext = os.path.splitext(base_lower)[0]

    # 1. Exact match on filename (with extension)
    if base_lower in _models_db:
        return (_enrich_info(_models_db[base_lower]), 1.0)

    # 2. Civitai mapping
    clean_name = name_no_ext.replace('-', '_').replace('.', '_')
    for map_key, map_path in (_civitai_map or {}).items():
        if map_key in clean_name or clean_name in map_key:
            # attempt to find the mapped path in DB entries
            mapped_basename = os.path.basename(map_path).lower()
            if mapped_basename in _models_db:
                return (_enrich_info(_models_db[mapped_basename]), 0.99)
            # fallback: return a constructed Kijai-style entry
            return ({
                'repo_id': 'Kijai/WanVideo_comfy',
                'path': map_path,
                'filename': os.path.basename(map_path),
                'url': f"https://huggingface.co/Kijai/WanVideo_comfy/resolve/main/{map_path}",
                'pageUrl': 'https://huggingface.co/Kijai/WanVideo_comfy'
            }, 0.99)

    # 3. Fuzzy matching using rapidfuzz if available
    try:
        from rapidfuzz import fuzz, process
        if _all_keys:
            result = process.extractOne(base_lower, _all_keys, scorer=fuzz.token_set_ratio)
            if result and result[1] >= 85:
                matched_key = result[0]
                return (_enrich_info(_models_db[matched_key]), result[1] / 100)
            result = process.extractOne(base_lower, _all_keys, scorer=fuzz.partial_ratio)
            if result and result[1] >= 90:
                matched_key = result[0]
                if len(matched_key) > len(base_lower) * 0.5:
                    return (_enrich_info(_models_db[matched_key]), result[1] / 100)
    except Exception:
        # fallback: simple difflib
        try:
            import difflib
            if _all_keys:
                matches = difflib.get_close_matches(name_no_ext, _all_keys, n=1, cutoff=0.85)
                if matches:
                    return (_enrich_info(_models_db[matches[0]]), 0.85)
        except Exception:
            pass

    return (None, 0)
