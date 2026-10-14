import sqlite3
import os
import json
import logging
import re
from datetime import datetime, timezone

logger = logging.getLogger(__name__)

try:
    from .config import get_matcher_config, get_features
except ImportError:
    from config import get_matcher_config, get_features

try:
    from .utils import AdvancedTokenizer
except ImportError:
    from utils import AdvancedTokenizer

try:
    from .mirror import rewrite_hf_url
except ImportError:
    from mirror import rewrite_hf_url

class ModelDatabase:
    """
    模型数据库管理器 (Phase 2)
    负责管理本地 SQLite 数据库，存储模型信息、哈希和别名。
    """
    def __init__(self, db_path=None):
        if db_path is None:
            # 默认存储在 core/data/models.db
            db_path = os.path.join(os.path.dirname(__file__), 'data', 'models.db')
        
        self.db_path = db_path
        self._ensure_db_dir()
        self._init_db()
        self._civitai_map = self._load_civitai_map()

    def _ensure_db_dir(self):
        """确保数据库目录存在"""
        directory = os.path.dirname(self.db_path)
        if not os.path.exists(directory):
            os.makedirs(directory)

    def _get_connection(self):
        """获取数据库连接"""
        return sqlite3.connect(self.db_path)

    def _init_db(self):
        """初始化数据库：运行可重放迁移到最新 schema 版本（T1.5）。"""
        self.run_migrations()

    def run_migrations(self):
        """应用所有待执行、有序、幂等的迁移。

        已应用的版本记录在 ``schema_migrations`` 表中，使 schema 始终可复现、
        可重放（T1.5 SQLite 迁移纪律）。
        """
        conn = self._get_connection()
        try:
            conn.execute('''
            CREATE TABLE IF NOT EXISTS schema_migrations (
                version INTEGER PRIMARY KEY,
                applied_at TEXT NOT NULL,
                description TEXT
            )
            ''')
            applied = {row[0] for row in conn.execute('SELECT version FROM schema_migrations')}
            for version, description, fn in MIGRATIONS:
                if version in applied:
                    continue
                fn(conn)
                conn.execute(
                    'INSERT INTO schema_migrations (version, applied_at, description) '
                    'VALUES (?, ?, ?)',
                    (version, datetime.now(timezone.utc).isoformat(), description),
                )
                conn.commit()
                logger.info('[DB] 应用迁移 v%d: %s', version, description)
        finally:
            conn.close()

    def get_schema_version(self):
        """返回已应用的最高迁移版本（无则为 0）。"""
        conn = self._get_connection()
        try:
            row = conn.execute('SELECT MAX(version) FROM schema_migrations').fetchone()
            return row[0] if row and row[0] is not None else 0
        except Exception:
            return 0
        finally:
            conn.close()

    def migrate(self):
        """公开入口：将数据库升级到最新 schema 版本。

        可重复调用。返回最终的 schema 版本号。
        """
        self.run_migrations()
        return self.get_schema_version()

    def _load_civitai_map(self):
        """Load CIVITAI_MAP from models_db.json so the matcher's SQLite path
        can resolve Civitai alias names exactly like the searcher's JSON path."""
        json_path = os.path.join(os.path.dirname(self.db_path), 'models_db.json')
        if not os.path.exists(json_path):
            return {}
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                payload = json.load(f)
            return payload.get('CIVITAI_MAP', {}) or {}
        except Exception:
            return {}

    def add_model(self, name, model_type=None, base_model=None, description=None):
        """添加或获取模型 ID"""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        try:
            cursor.execute('''
            INSERT OR IGNORE INTO models (name, type, base_model, description)
            VALUES (?, ?, ?, ?)
            ''', (name, model_type, base_model, description))
            conn.commit()
            
            # 获取 ID (无论是新插入的还是已存在的)
            cursor.execute('SELECT id FROM models WHERE name = ?', (name,))
            row = cursor.fetchone()
            return row[0] if row else None
        finally:
            conn.close()

    def add_hash(self, sha256, model_id, filename=None, source="User"):
        """添加文件哈希映射"""
        conn = self._get_connection()
        cursor = conn.cursor()
        try:
            cursor.execute('''
            INSERT OR REPLACE INTO file_hashes (hash_sha256, model_id, filename, source)
            VALUES (?, ?, ?, ?)
            ''', (sha256, model_id, filename, source))
            conn.commit()
        finally:
            conn.close()

    def add_alias(self, alias, model_id, is_regex=False):
        """添加模型别名"""
        conn = self._get_connection()
        cursor = conn.cursor()
        try:
            cursor.execute('''
            INSERT OR REPLACE INTO aliases (alias, model_id, is_regex)
            VALUES (?, ?, ?)
            ''', (alias.lower(), model_id, is_regex))
            conn.commit()
        finally:
            conn.close()

    def get_model_by_hash(self, sha256):
        """通过哈希查找模型"""
        conn = self._get_connection()
        cursor = conn.cursor()
        try:
            cursor.execute('''
            SELECT m.name, m.type, m.base_model, h.source 
            FROM file_hashes h
            JOIN models m ON h.model_id = m.id
            WHERE h.hash_sha256 = ?
            ''', (sha256,))
            return cursor.fetchone()
        finally:
            conn.close()

    def search_by_filename(self, filename):
        """
        通过文件名查找模型 (Phase 2 Simple Implementation)
        目前主要匹配 aliases 表中的精确别名
        返回: (name, type, base_model, description)
        """
        conn = self._get_connection()
        cursor = conn.cursor()
        filename_lower = filename.lower()
        try:
            # 尝试直接匹配 alias (需移除扩展名)
            base_name = os.path.splitext(filename_lower)[0]
            
            cursor.execute('''
            SELECT m.name, m.type, m.base_model, m.description
            FROM aliases a
            JOIN models m ON a.model_id = m.id
            WHERE a.alias = ? AND a.is_regex = 0
            ''', (base_name,))
            
            result = cursor.fetchone()
            if result:
                return result
                
            # TODO: 实现正则匹配 (L3) 和 模糊匹配逻辑
            return None
        finally:
            conn.close()

# ... existing code ...

    def populate_from_json(self):
        """从 models_data.json 迁移数据到数据库 (Phase 2 Migration)"""
        json_path = os.path.join(os.path.dirname(self.db_path), 'models_data.json')
        if not os.path.exists(json_path):
            logger.warning(f"[DB] JSON data file not found: {json_path}")
            return

        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except Exception as e:
            logger.exception(f"[DB] Error loading JSON: {e}")
            return

        conn = self._get_connection()
        cursor = conn.cursor()
        
        added_models = 0
        added_aliases = 0
        
        try:
            # 1. 迁移 Popular Models
            # 假设这些是 "标准模型"，尽管名字可能还是文件名格式，但在 Phase 3 我们会进一步清理
            popular_models = data.get('popular_models', {})
            for name, repo_id in popular_models.items():
                # 插入模型
                # 推断 base_model (简单的启发式)
                base_model = "Unknown"
                name_lower = name.lower()
                if "sdxl" in name_lower: base_model = "SDXL"
                elif "flux" in name_lower: base_model = "Flux"
                elif "sd1.5" in name_lower or "v1-5" in name_lower: base_model = "SD1.5"
                elif "sd3" in name_lower: base_model = "SD3"
                elif "wan" in name_lower: base_model = "Wan"

                cursor.execute('''
                INSERT OR IGNORE INTO models (name, type, base_model, description)
                VALUES (?, ?, ?, ?)
                ''', (name, "Checkpoint", base_model, f"Repo: {repo_id}"))
                
                # 获取 ID
                cursor.execute('SELECT id FROM models WHERE name = ?', (name,))
                row = cursor.fetchone()
                if row:
                    model_id = row[0]
                    # 添加 Alias (即它自己)
                    cursor.execute('INSERT OR IGNORE INTO aliases (alias, model_id) VALUES (?, ?)', (name.lower(), model_id))
                    # 添加 Hash (暂时没有 SHA256，但我们可以把 Repo ID 存到 description 或其他地方，这里先跳过 file_hashes，因为没有 hash)
                    added_models += 1

            # 2. 迁移 Model Aliases (作为虚拟模型或关联到现有模型?)
            # 目前 model_aliases 只是 缩写 -> 全名 的映射 (e.g. 'sdxl' -> 'stable diffusion xl')
            # 这实际上不应该存为 'models' 表的一行，而是应该作为 全局别名/同义词
            # 但为了 DB 完整性，我们可以把 'Full Name' 存为模型，把 'Abbr' 存为 alias
            aliases_map = data.get('model_aliases', {})
            for abbr, full_name in aliases_map.items():
                # 插入全名作为模型
                cursor.execute('''
                INSERT OR IGNORE INTO models (name, type, base_model, description)
                VALUES (?, ?, ?, ?)
                ''', (full_name, "Concept", "Unknown", "General Concept"))
                
                cursor.execute('SELECT id FROM models WHERE name = ?', (full_name,))
                row = cursor.fetchone()
                if row:
                    model_id = row[0]
                    # 插入缩写作为 alias
                    cursor.execute('INSERT OR IGNORE INTO aliases (alias, model_id) VALUES (?, ?)', (abbr.lower(), model_id))
                    added_aliases += 1

            conn.commit()
            logger.info(f"[DB] Migration complete. Added {added_models} models and {added_aliases} aliases.")
            
        except Exception as e:
            logger.exception(f"[DB] Migration failed: {e}")
            conn.rollback()
        finally:
            conn.close()

    def import_models_db_json(self, json_path=None):
        """Import core/data/models_db.json into external_models table."""
        if json_path is None:
            json_path = os.path.join(os.path.dirname(self.db_path), 'models_db.json')
        if not os.path.exists(json_path):
            logger.warning(f"[DB] models_db.json not found: {json_path}")
            return 0

        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                payload = json.load(f)
        except Exception as e:
            logger.exception(f"[DB] Error loading models_db.json: {e}")
            return 0

        models = payload.get('MODELS_DB', {})
        if not models:
            logger.warning("[DB] No MODELS_DB section in JSON")
            return 0

        conn = self._get_connection()
        cursor = conn.cursor()
        inserted = 0
        try:
            for key, info in models.items():
                filename_lower = key
                repo_id = info.get('repo_id')
                path = info.get('path')
                filename = info.get('filename')
                source = info.get('source')

                # infer basic semantic fields
                inferred_type = self._infer_type_from_filename(filename or key, repo_id or '')
                inferred_base_model = self._infer_base_model(filename or key, repo_id or '')
                inferred_family = self._infer_family(filename or key, repo_id or '')
                normalized_name = self._normalize_model_name(filename or key)
                alias = self._extract_alias(filename or key)
                tokens = self._build_tokens(filename or key)

                cursor.execute('''
                INSERT OR REPLACE INTO external_models (
                    filename_lower, repo_id, path, filename, source,
                    normalized_name, alias, type, base_model, family, tokens
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    filename_lower,
                    repo_id,
                    path,
                    filename,
                    source,
                    normalized_name,
                    alias,
                    inferred_type,
                    inferred_base_model,
                    inferred_family,
                    tokens,
                ))
                inserted += 1
            conn.commit()
            logger.info(f"[DB] Imported {inserted} external_models from {json_path}")
        except Exception as e:
            logger.exception(f"[DB] Import failed: {e}")
            conn.rollback()
        finally:
            conn.close()
        return inserted

    def lookup_modelsdb(self, filename, expected_types=None):
        """Lookup filename in external_models table.
        Returns (info_dict, score) or (None, 0)
        Strategy: exact filename_lower -> config score; basename match -> config score; semantic token overlap -> config threshold; fuzzy fallback
        """
        cfg = get_matcher_config().get('db', {})
        if not cfg.get('enabled', True):
            return (None, 0)

        if not filename:
            return (None, 0)
        base = os.path.basename(filename)
        base_lower = base.lower()
        base_no_ext = os.path.splitext(base_lower)[0]

        target_tokens = self._tokenize_text(base_no_ext)
        target_alias = self._extract_alias(base_no_ext)
        target_normalized = self._normalize_search_text(base_no_ext)

        conn = self._get_connection()
        cursor = conn.cursor()
        try:
            cursor.execute('''
            SELECT repo_id, path, filename, source, normalized_name, alias, type, base_model, family, tokens
            FROM external_models WHERE filename_lower = ?
            ''', (base_lower,))
            row = cursor.fetchone()
            if row:
                info = self._row_to_info(row)
                if self._type_ok(info.get('type'), expected_types):
                    return (_enrich_external_info(info), cfg['exact_score'])

            cursor.execute('''
            SELECT repo_id, path, filename, source, normalized_name, alias, type, base_model, family, tokens
            FROM external_models WHERE filename_lower = ?
            ''', (base_no_ext,))
            row = cursor.fetchone()
            if row:
                info = self._row_to_info(row)
                if self._type_ok(info.get('type'), expected_types):
                    return (_enrich_external_info(info), cfg['basename_score'])

            # Civitai alias resolution (mirrors searcher.find_best_match_in_db so
            # the matcher's SQLite path and the searcher's JSON path stay consistent)
            if self._civitai_map:
                clean_name = base_no_ext.replace('-', '_').replace('.', '_')
                for map_key, map_path in self._civitai_map.items():
                    if map_key in clean_name or clean_name in map_key:
                        mapped_basename = os.path.basename(map_path).lower()
                        cursor.execute('''
                        SELECT repo_id, path, filename, source, normalized_name, alias, type, base_model, family, tokens
                        FROM external_models WHERE filename_lower = ?
                        ''', (mapped_basename,))
                        mrow = cursor.fetchone()
                        if mrow:
                            minfo = self._row_to_info(mrow)
                            if self._type_ok(minfo.get('type'), expected_types):
                                return (_enrich_external_info(minfo), cfg['civitai_score'])
                        return ({
                            'repo_id': 'Kijai/WanVideo_comfy',
                            'path': map_path,
                            'filename': os.path.basename(map_path),
                            'url': rewrite_hf_url(f"https://huggingface.co/Kijai/WanVideo_comfy/resolve/main/{map_path}"),
                            'pageUrl': 'https://huggingface.co/Kijai/WanVideo_comfy'
                        }, cfg['civitai_score'])

            cursor.execute('''
            SELECT repo_id, path, filename, source, normalized_name, alias, type, base_model, family, tokens
            FROM external_models
            ''')
            rows = cursor.fetchall()
        finally:
            conn.close()

        best_info = None
        best_score = 0.0
        for row in rows or []:
            info = self._row_to_info(row)
            if expected_types and info.get('type') not in expected_types and info.get('type') != 'unknown':
                continue

            score = 0.0
            if target_alias and info.get('alias') and target_alias == info.get('alias'):
                score += 0.25
            if info.get('normalized_name'):
                cand_tokens = self._tokenize_text(info.get('normalized_name'))
                if cand_tokens:
                    overlap = len(set(target_tokens) & set(cand_tokens))
                    union = len(set(target_tokens) | set(cand_tokens))
                    if union:
                        score += (overlap / union) * 0.6
            if info.get('tokens'):
                cand_tokens = self._tokenize_text(info.get('tokens'))
                if cand_tokens:
                    overlap = len(set(target_tokens) & set(cand_tokens))
                    union = len(set(target_tokens) | set(cand_tokens))
                    if union:
                        score += (overlap / union) * 0.2
            if target_normalized and info.get('filename'):
                candidate_normalized = self._normalize_search_text(info.get('filename'))
                if candidate_normalized:
                    if target_normalized == candidate_normalized:
                        score += 0.2
                    elif target_normalized in candidate_normalized or candidate_normalized in target_normalized:
                        score += 0.15
            if info.get('base_model') and info.get('base_model') != 'Unknown':
                if target_tokens and info.get('base_model').lower() in ' '.join(target_tokens):
                    score += 0.1

            # [T2.2] 版本感知降权：同族不同版本强降权（gated by features.version_aware）
            try:
                if get_features().get('version_aware', False):
                    fam_t, maj_t, min_t = AdvancedTokenizer.parse_version_tuple(base_no_ext)
                    fam_c, maj_c, min_c = AdvancedTokenizer.parse_version_tuple(info.get('filename') or '')
                    if fam_t and fam_c and fam_t == fam_c:
                        if (maj_t is not None and maj_c is not None and maj_t != maj_c) or \
                           (min_t is not None and min_c is not None and min_t != min_c):
                            score *= 0.3
            except Exception:
                pass

            if score > best_score and score >= cfg['semantic_min_score']:
                best_score = score
                best_info = info

        if best_info:
            return (_enrich_external_info(best_info), best_score)

        try:
            from rapidfuzz import fuzz, process
            conn = self._get_connection()
            cursor = conn.cursor()
            try:
                cursor.execute('SELECT filename_lower FROM external_models')
                keys = [r[0] for r in cursor.fetchall()]
                if keys:
                    res = process.extractOne(base_lower, keys, scorer=fuzz.token_set_ratio)
                    if res and res[1] >= cfg['fuzzy_score_cutoff']:
                        matched_key = res[0]
                        cursor.execute('''
                        SELECT repo_id, path, filename, source, normalized_name, alias, type, base_model, family, tokens
                        FROM external_models WHERE filename_lower = ?
                        ''', (matched_key,))
                        row = cursor.fetchone()
                        if row:
                            info = self._row_to_info(row)
                            if not expected_types or info.get('type') in expected_types or info.get('type') == 'unknown':
                                return (_enrich_external_info(info), res[1] / 100)
            finally:
                conn.close()
        except Exception:
            pass

        return (None, 0)

    def _tokenize_text(self, value):
        if not value:
            return set()
        tokenized = []
        text = os.path.basename(str(value))
        text = text.replace('_', ' ').replace('-', ' ').replace('.', ' ')
        for part in re.split(r'\s+', text.lower()):
            if part:
                tokenized.append(part)
        return set(tokenized)

    def _normalize_search_text(self, value):
        if not value:
            return ''
        text = os.path.basename(str(value)).lower()
        text = re.sub(r'[_\-\.\s]+', '', text)
        return text

    def _row_to_info(self, row):
        return {
            'repo_id': row[0],
            'path': row[1],
            'filename': row[2],
            'source': row[3],
            'normalized_name': row[4],
            'alias': row[5],
            'type': row[6],
            'base_model': row[7],
            'family': row[8],
            'tokens': row[9],
        }

    def _type_ok(self, model_type, expected_types):
        """Return True if model_type passes the expected_types filter.
        'unknown' type is always allowed (best-effort), matching matcher semantics.
        """
        if not expected_types:
            return True
        if model_type == 'unknown':
            return True
        return model_type in expected_types

    def _infer_type_from_filename(self, filename, repo_id=''):
        lower = (filename or '').lower()
        if 'lora' in lower or 'loras' in lower:
            return 'lora'
        if 'vae' in lower:
            return 'vae'
        if 'controlnet' in lower:
            return 'controlnet'
        if 'upscaler' in lower or 'esrgan' in lower or 'swinir' in lower or 'upscale' in lower:
            return 'upscale_model'
        if 'clip' in lower or 'text_encoder' in lower:
            return 'clip'
        if 'unet' in lower:
            return 'unet'
        if 'embedding' in lower:
            return 'embeddings'
        if 'gguf' in lower:
            return 'checkpoint'
        return 'checkpoint'

    def _infer_base_model(self, filename, repo_id=''):
        lower = (filename or '').lower()
        if 'sdxl' in lower or 'stable diffusion xl' in lower:
            return 'SDXL'
        if 'sd3' in lower:
            return 'SD3'
        if 'flux' in lower:
            return 'Flux'
        if 'wan' in lower:
            return 'Wan'
        if 'qwen' in lower:
            return 'Qwen'
        if 'pony' in lower:
            return 'Pony'
        if 'hunyuan' in lower:
            return 'Hunyuan'
        return 'Unknown'

    def _infer_family(self, filename, repo_id=''):
        lower = (filename or '').lower()
        if 'wan' in lower:
            return 'Wan'
        if 'flux' in lower:
            return 'Flux'
        if 'qwen' in lower:
            return 'Qwen'
        if 'sdxl' in lower:
            return 'SDXL'
        if 'sd3' in lower:
            return 'SD3'
        if 'pony' in lower:
            return 'Pony'
        if 'hunyuan' in lower:
            return 'Hunyuan'
        return 'General'

    def _normalize_model_name(self, value):
        name = os.path.basename(value or '')
        name = os.path.splitext(name)[0]
        name = name.replace('_', ' ').replace('-', ' ').replace('.', ' ')
        name = re.sub(r'\s+', ' ', name).strip().lower()
        return name

    def _extract_alias(self, value):
        name = os.path.basename(value or '')
        base = os.path.splitext(name)[0].lower()
        if base.startswith('wan2_1'):
            return 'wan2.1'
        if base.startswith('wan2_2'):
            return 'wan2.2'
        if 'flux1' in base:
            return 'flux1'
        if 'sdxl' in base:
            return 'sdxl'
        return None

    def _build_tokens(self, value):
        name = os.path.basename(value or '')
        base = os.path.splitext(name)[0]
        tokens = []
        for part in re.split(r'[_\-\.\s]+', base.lower()):
            if part:
                tokens.append(part)
        return ' '.join(tokens)


def _enrich_external_info(info):
    new_info = info.copy()
    repo_id = info.get('repo_id')
    path = info.get('path')
    if repo_id and path:
        new_info['url'] = rewrite_hf_url(f"https://huggingface.co/{repo_id}/resolve/main/{path}")
        new_info['pageUrl'] = f"https://huggingface.co/{repo_id}/tree/main"
    return new_info

# ---------------------------------------------------------------------------
# Schema 迁移（T1.5）：显式、有序、幂等、可重放。
# SCHEMA_VERSION 为最新版本号；新增迁移时同步递增。
# 每个迁移函数接收一个已打开的 sqlite3 连接，必须可重复执行
# （使用 IF NOT EXISTS / PRAGMA 守卫）。
# ---------------------------------------------------------------------------
SCHEMA_VERSION = 2


def _migration_v1(conn):
    """基础 schema：models / file_hashes / aliases / external_models（核心列）。"""
    cursor = conn.cursor()
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS models (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT NOT NULL UNIQUE,
        type TEXT,
        base_model TEXT,
        description TEXT
    )
    ''')
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS file_hashes (
        hash_sha256 TEXT PRIMARY KEY,
        model_id INTEGER,
        filename TEXT,
        source TEXT,
        FOREIGN KEY(model_id) REFERENCES models(id)
    )
    ''')
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS aliases (
        alias TEXT PRIMARY KEY,
        model_id INTEGER,
        is_regex BOOLEAN DEFAULT 0,
        FOREIGN KEY(model_id) REFERENCES models(id)
    )
    ''')
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS external_models (
        filename_lower TEXT PRIMARY KEY,
        repo_id TEXT,
        path TEXT,
        filename TEXT,
        source TEXT
    )
    ''')
    cursor.execute('CREATE INDEX IF NOT EXISTS ix_external_models_filename '
                   'ON external_models(filename_lower)')
    conn.commit()


def _migration_v2(conn):
    """为 external_models 增加语义列（带列存在性守卫，幂等）。"""
    cursor = conn.cursor()
    existing = {row[1] for row in cursor.execute('PRAGMA table_info(external_models)')}
    for col in ('normalized_name', 'alias', 'type', 'base_model', 'family', 'tokens'):
        if col not in existing:
            cursor.execute(f'ALTER TABLE external_models ADD COLUMN {col} TEXT')
    cursor.execute('CREATE INDEX IF NOT EXISTS ix_external_models_normalized '
                   'ON external_models(normalized_name)')
    cursor.execute('CREATE INDEX IF NOT EXISTS ix_external_models_alias '
                   'ON external_models(alias)')
    cursor.execute('CREATE INDEX IF NOT EXISTS ix_external_models_type '
                   'ON external_models(type)')
    conn.commit()


MIGRATIONS = [
    (1, 'create base schema (models/file_hashes/aliases/external_models)', _migration_v1),
    (2, 'add semantic columns to external_models (normalized_name/alias/type/base_model/family/tokens)', _migration_v2),
]


# 单例实例
db = ModelDatabase()
if __name__ == "__main__":
    # 如果直接运行脚本，执行迁移并打印当前 schema 版本
    version = db.migrate()
    print(f"[DB] 迁移完成，当前 schema 版本: v{version}")
