"""T2.2 版本/族谱感知匹配回归测试。

验证：
  1. parse_version_tuple 正确抽取 (family, major, minor)
  2. calculate_similarity 在 version_aware 开启时对同族不同版本强降权
  3. matcher._check_conflicts 对同族不同版本 / flux dev↔schnell / sdxl base↔refiner 判硬冲突
  4. 端到端：wan2.1 输入不应误配 wan2.2；flux dev 不应误配 schnell；同版本应正常匹配
  5. 生产开关 core/data/matcher_config.json 的 features.version_aware == true

遵循 migration_check.py 的隔离范式：stub folder_paths，使用托管 venv 运行。
"""
import os
import sys
import json
import unittest
from unittest.mock import MagicMock

# 隔离 stub：core 模块在导入链中可能触及 folder_paths / server
sys.modules['folder_paths'] = MagicMock()

import types as _types
_server_stub = _types.ModuleType("server")
_server_ps = MagicMock()
_server_routes = MagicMock()


def _identity_decorator(_path):
    def _wrap(fn):
        return fn
    return _wrap


_server_routes.post.side_effect = _identity_decorator
_server_routes.get.side_effect = _identity_decorator
_server_ps.instance.routes = _server_routes
_server_stub.PromptServer = _server_ps
sys.modules['server'] = _server_stub

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'core'))

from core import config as config_module
from core.utils import AdvancedTokenizer
from core.matcher import ModelMatcher


def setUpModule():
    # 确保 T2.2 开关在测试期间生效；测试结束还原缓存
    config_module.override_config({
        'features': {'version_aware': True},
        'matching': {'use_db_fallback': False},  # 隔离到内存匹配路径
    })


def tearDownModule():
    config_module.reset_config()


class TestParseVersionTuple(unittest.TestCase):
    def _check(self, name, expected_family, expected_major=None, expected_minor=None):
        fam, maj, min_ = AdvancedTokenizer.parse_version_tuple(name)
        self.assertEqual(fam, expected_family, f"{name}: family {fam} != {expected_family}")
        if expected_major is not None:
            self.assertEqual(maj, expected_major, f"{name}: major {maj} != {expected_major}")
        if expected_minor is not None:
            self.assertEqual(min_, expected_minor, f"{name}: minor {min_} != {expected_minor}")

    def test_wan(self):
        self._check("wan2.1-t2v-14b.safetensors", 'wan', 2, 1)
        self._check("wan2_2-i2v-14b.safetensors", 'wan', 2, 2)
        self._check("Wan21_T2V_14B.safetensors", 'wan', 2, 1)

    def test_flux(self):
        self._check("flux1-dev.safetensors", 'flux', 1, None)
        self._check("flux.1-schnell.safetensors", 'flux', 1, None)
        self._check("flux2-dev.safetensors", 'flux', 2, None)

    def test_sdxl(self):
        self._check("sd_xl_base_1.0.safetensors", 'sdxl', 1, 0)
        self._check("SDXL_Juggernaut_XL_v9.safetensors", 'sdxl', None, None)

    def test_sd15(self):
        self._check("v1-5-pruned-emaonly.ckpt", 'sd15', 1, 5)
        self._check("sd15.safetensors", 'sd15', 1, 5)

    def test_unknown(self):
        fam, maj, min_ = AdvancedTokenizer.parse_version_tuple("my_random_model.safetensors")
        self.assertIsNone(fam)


class TestSimilarityVersionAware(unittest.TestCase):
    def _score(self, a, b, aware):
        config_module.override_config({
            'features': {'version_aware': aware},
            'matching': {'use_db_fallback': False},
        })
        try:
            return AdvancedTokenizer.calculate_similarity(a, b)
        finally:
            config_module.reset_config()

    def test_wan21_vs_wan22_downweighted(self):
        """同族不同版本：开启后被强降权（*0.3），且低于 searcher 阈值 0.35。"""
        off = self._score("wan2.1-t2v-14b.safetensors", "wan2.2-t2v-14b.safetensors", False)
        on = self._score("wan2.1-t2v-14b.safetensors", "wan2.2-t2v-14b.safetensors", True)
        self.assertGreater(off, 0.0, "关闭时同族仍有基础相似度")
        self.assertLess(on, 0.35, "开启后 wan2.1 vs wan2.2 应低于阈值")
        self.assertAlmostEqual(on, off * 0.3, delta=1e-6, msg="开启后应被乘以 0.3")

    def test_wan21_same_version_not_penalized(self):
        """同族同版本（仅 i2v/t2v 差异）：不应被版本逻辑降权。"""
        off = self._score("wan2.1-i2v-14b.safetensors", "wan2.1-t2v-14b.safetensors", False)
        on = self._score("wan2.1-i2v-14b.safetensors", "wan2.1-t2v-14b.safetensors", True)
        self.assertGreater(on, 0.35, "wan2.1 同版本应仍匹配")
        self.assertAlmostEqual(on, off, delta=1e-6, msg="同版本不应被降权")

    def test_flux_dev_vs_schnell_zero(self):
        """flux dev vs schnell：硬零（族谱变体冲突）。"""
        on = self._score("flux1-dev.safetensors", "flux1-schnell.safetensors", True)
        self.assertEqual(on, 0.0, "flux dev 不应匹配 schnell")

    def test_sdxl_base_vs_refiner_zero(self):
        on = self._score("sd_xl_base_1.0.safetensors", "sd_xl_refiner_1.0.safetensors", True)
        self.assertEqual(on, 0.0, "sdxl base 不应匹配 refiner")

    def test_sd15_same_not_penalized(self):
        """sd1.5 同版本（v1-5 与 sd1.5 命名差异）：版本逻辑不降权。"""
        off = self._score("v1-5-pruned-emaonly.safetensors", "sd1.5-pruned-emaonly.safetensors", False)
        on = self._score("v1-5-pruned-emaonly.safetensors", "sd1.5-pruned-emaonly.safetensors", True)
        self.assertGreater(on, 0.2, "sd1.5 同版本应仍匹配")
        self.assertAlmostEqual(on, off, delta=1e-6, msg="sd1.5 同版本不应被降权")

    def test_cross_family_unaffected(self):
        """跨族不受版本逻辑影响（仅架构/量化冲突才会归零）。"""
        on = self._score("ghostmix_v1.safetensors", "ghostmix-v1.0-pruned.safetensors", True)
        self.assertGreater(on, 0.4)


class TestMatcherVersionConflict(unittest.TestCase):
    def setUp(self):
        config_module.override_config({
            'features': {'version_aware': True},
            'matching': {'use_db_fallback': False},
        })

    def tearDown(self):
        config_module.reset_config()

    def _make_matcher(self, local_models):
        scanner = MagicMock()
        scanner.get_all_models.return_value = local_models
        matcher = ModelMatcher(scanner)
        matcher._build_index()
        return matcher

    def test_wan21_not_matched_to_wan22(self):
        matcher = self._make_matcher([
            {"filename": "wan2.2-t2v-14b.safetensors", "path": "/x/wan2.2-t2v-14b.safetensors", "type": "checkpoints"},
        ])
        items = [{"id": 1, "current": "wan2.1-t2v-14b.safetensors",
                  "node_type": "CheckpointLoaderSimple", "widget_name": "ckpt_name"}]
        result = matcher.match(items)
        self.assertEqual(len(result), 0, "wan2.1 不应误配 wan2.2")

    def test_wan21_matched_to_wan21(self):
        matcher = self._make_matcher([
            {"filename": "wan2.1-t2v-14b.safetensors", "path": "/x/wan2.1-t2v-14b.safetensors", "type": "checkpoints"},
        ])
        items = [{"id": 1, "current": "wan2.1-t2v-14b.safetensors",
                  "node_type": "CheckpointLoaderSimple", "widget_name": "ckpt_name"}]
        result = matcher.match(items)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["matched_value"], "wan2.1-t2v-14b.safetensors")

    def test_flux_dev_not_matched_to_schnell(self):
        matcher = self._make_matcher([
            {"filename": "flux1-schnell.safetensors", "path": "/x/flux1-schnell.safetensors", "type": "checkpoints"},
        ])
        items = [{"id": 2, "current": "flux1-dev.safetensors",
                  "node_type": "CheckpointLoaderSimple", "widget_name": "ckpt_name"}]
        result = matcher.match(items)
        self.assertEqual(len(result), 0, "flux dev 不应误配 schnell")

    def test_conflict_helper(self):
        matcher = self._make_matcher([])
        self.assertTrue(matcher._version_family_conflict("wan2.1-a.safetensors", "wan2.2-a.safetensors"))
        self.assertFalse(matcher._version_family_conflict("wan2.1-a.safetensors", "wan2.1-b.safetensors"))
        self.assertTrue(matcher._version_family_conflict("flux1-dev.safetensors", "flux1-schnell.safetensors"))


class TestProductionFlag(unittest.TestCase):
    def test_version_aware_enabled_in_config(self):
        cfg_path = os.path.join(ROOT, 'core', 'data', 'matcher_config.json')
        with open(cfg_path, 'r', encoding='utf-8') as f:
            cfg = json.load(f)
        self.assertTrue(
            cfg.get('features', {}).get('version_aware', False),
            "生产配置 core/data/matcher_config.json 必须开启 features.version_aware")


if __name__ == '__main__':
    unittest.main(verbosity=2)
