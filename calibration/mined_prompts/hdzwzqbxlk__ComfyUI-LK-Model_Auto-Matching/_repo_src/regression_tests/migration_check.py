"""迁移纪律回归测试（T1.5）。

验证：
  1. 全新数据库迁移到最新 SCHEMA_VERSION。
  2. run_migrations() 幂等（可重放）—— 重复执行无错、版本稳定。
  3. 遗留数据库（external_models 缺 v2 语义列）可被安全升级且数据不丢。
  4. MIGRATIONS 最高版本与 SCHEMA_VERSION 常量一致。

运行：
    python regression_tests/migration_check.py
"""
import os
import sqlite3
import sys
import tempfile
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# Stub ComfyUI's `folder_paths` module (imported transitively via core/__init__.py
# -> core.scanner). A MagicMock is enough for the migration logic under test.
import types  # noqa: E402
from unittest.mock import MagicMock  # noqa: E402
_folder_paths_stub = types.ModuleType("folder_paths")
_folder_paths_stub.get_folder_paths = lambda *a, **k: []
_folder_paths_stub.get_input_directory = lambda *a, **k: "mock_input_dir"
_folder_paths_stub.get_output_directory = lambda *a, **k: "mock_output_dir"
_folder_paths_stub.get_temp_directory = lambda *a, **k: "mock_temp_dir"
sys.modules["folder_paths"] = _folder_paths_stub

from core.database import ModelDatabase, SCHEMA_VERSION, MIGRATIONS  # noqa: E402


def _new_temp_db():
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    os.remove(path)  # 让 sqlite 重新创建全新库
    return path


class MigrationTests(unittest.TestCase):
    def test_fresh_db_migrates_to_latest(self):
        path = _new_temp_db()
        try:
            db = ModelDatabase(db_path=path)
            self.assertEqual(db.get_schema_version(), SCHEMA_VERSION)
        finally:
            os.remove(path)

    def test_migrations_are_idempotent(self):
        path = _new_temp_db()
        try:
            db = ModelDatabase(db_path=path)
            v1 = db.get_schema_version()
            # 显式重放多次
            db.run_migrations()
            db.run_migrations()
            self.assertEqual(db.get_schema_version(), v1)
            self.assertEqual(v1, SCHEMA_VERSION)
        finally:
            os.remove(path)

    def test_legacy_db_upgraded_safely(self):
        path = _new_temp_db()
        try:
            # 模拟 v2 之前的库：external_models 不含语义列，但已有数据
            seed = sqlite3.connect(path)
            seed.execute('''CREATE TABLE external_models (
                filename_lower TEXT PRIMARY KEY,
                repo_id TEXT, path TEXT, filename TEXT, source TEXT)''')
            seed.execute(
                "INSERT INTO external_models VALUES ("
                "'wan2.1_t2v_14b.safetensors', 'Kijai/WanVideo_comfy', "
                "'wan2.1_t2v_14B.safetensors', 'wan2.1_t2v_14B.safetensors', 'Comfy-Org')")
            seed.commit()
            seed.close()

            db = ModelDatabase(db_path=path)  # 构造时触发 run_migrations
            self.assertEqual(db.get_schema_version(), SCHEMA_VERSION)

            conn = db._get_connection()
            cols = {r[1] for r in conn.execute('PRAGMA table_info(external_models)')}
            conn.close()
            for col in ('normalized_name', 'alias', 'type', 'base_model', 'family', 'tokens'):
                self.assertIn(col, cols)

            # 数据必须保留
            conn = db._get_connection()
            row = conn.execute(
                'SELECT repo_id FROM external_models WHERE filename_lower=?',
                ('wan2.1_t2v_14b.safetensors',)).fetchone()
            conn.close()
            self.assertIsNotNone(row)
            self.assertEqual(row[0], 'Kijai/WanVideo_comfy')
        finally:
            os.remove(path)

    def test_migration_count_matches_constant(self):
        self.assertEqual(max(v for v, _, _ in MIGRATIONS), SCHEMA_VERSION)


if __name__ == "__main__":
    unittest.main()
