"""针对「本地核心词覆盖率误匹配 + 在线组件过滤/国内镜像优先级」Bug 的回归测试。

对应修复：
- matcher.py 新增核心身份词覆盖率硬门槛，技术后缀（fp16 等）不可单独支撑匹配；
- searcher.py 新增组件类别过滤（主模型请求剔除 text_encoder/vae/clip/dav 候选）
  与源优先级加权（主模型优先 HuggingFace / ModelScope / CNB 等官方国内镜像）。
"""
import sys
import os
import unittest
from unittest.mock import MagicMock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.modules['folder_paths'] = MagicMock()

from core.matcher import ModelMatcher
from core.searcher import ModelSearcher


class TestLocalCoreCoverage(unittest.TestCase):
    """本地匹配：核心身份词几乎不重叠时必须拒绝，正常变体仍匹配。"""

    def _make_matcher(self, models):
        scanner = MagicMock()
        scanner.get_all_models.return_value = models
        matcher = ModelMatcher(scanner)
        matcher._build_index()
        return matcher

    def test_minimax_must_not_match_sam(self):
        """minimax_music3_dit 不应被误匹配到 sam3.1.multiplex（仅 fp16 后缀相同）。"""
        models = [
            # 干扰项：目录 + 全异核心身份词
            {"filename": "00_Tools_C/sam3.1.multiplex_fp16.safetensors",
             "path": "D:/models/00_Tools_C/sam3.1.multiplex_fp16.safetensors",
             "type": "checkpoints"},
            # 其它完全无关的本地文件，确保不是「无候选」导致的空结果
            {"filename": "sd_xl_base_1.0.safetensors",
             "path": "D:/models/sd_xl_base_1.0.safetensors",
             "type": "checkpoints"},
            {"filename": "anything_v3.safetensors",
             "path": "D:/models/anything_v3.safetensors",
             "type": "checkpoints"},
        ]
        matcher = self._make_matcher(models)
        items = [{
            "id": 1,
            "current": "minimax_music3_dit_fp16.safetensors",
            "node_type": "ModelLoader",
            "widget_name": "diffusion_model",
        }]
        result = matcher.match(items)
        self.assertEqual(len(result), 0,
                         "minimax_music3_dit 不应匹配任何本地文件（核心词未覆盖）")

    def test_minimax_variant_still_matches(self):
        """若本地存在真正同族变体（minimax_music3_dit_bf16），应正常匹配。"""
        models = [
            {"filename": "00_Tools_C/sam3.1.multiplex_fp16.safetensors",
             "path": "D:/models/00_Tools_C/sam3.1.multiplex_fp16.safetensors",
             "type": "checkpoints"},
            {"filename": "minimax_music3_dit_bf16.safetensors",
             "path": "D:/models/minimax_music3_dit_bf16.safetensors",
             "type": "checkpoints"},
        ]
        matcher = self._make_matcher(models)
        items = [{
            "id": 1,
            "current": "minimax_music3_dit_fp16.safetensors",
            "node_type": "ModelLoader",
            "widget_name": "diffusion_model",
        }]
        result = matcher.match(items)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["matched_value"],
                         "minimax_music3_dit_bf16.safetensors")

    def test_flux_variant_still_matches(self):
        """回归：flux1-dev 与 flux1-dev-fp8 属正常变体，不应被覆盖率门槛误杀。"""
        models = [
            {"filename": "flux1-dev-fp8.safetensors",
             "path": "D:/models/flux1-dev-fp8.safetensors",
             "type": "unet"},
        ]
        matcher = self._make_matcher(models)
        items = [{
            "id": 1,
            "current": "flux1-dev.safetensors",
            "node_type": "UnetLoader",
            "widget_name": "unet_name",
        }]
        result = matcher.match(items)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["matched_value"], "flux1-dev-fp8.safetensors")

    def test_single_word_base_precision_variant_still_matches(self):
        """回归（QA 反馈的 latent 边角）：单字基本体 + 精度后缀（mycoolmodel_fp16）
        不应被覆盖率门槛误杀，应正常匹配其本体 mycoolmodel。

        根因：tokenize 把 fp16 拆成 fp + 数字，精度片段 fp 曾被算作核心身份词，
        导致 target_core={mycoolmodel, fp}（≥2）触发门槛、覆盖率仅 0.5<0.6 被拒。
        修复后先剥离精度变体词再分词，fp 片段不再泄漏，单字基本体经变体匹配路径命中。
        """
        models = [
            {"filename": "mycoolmodel.safetensors",
             "path": "D:/models/mycoolmodel.safetensors",
             "type": "checkpoints"},
        ]
        matcher = self._make_matcher(models)
        items = [{
            "id": 1,
            "current": "mycoolmodel_fp16.safetensors",
            "node_type": "ModelLoader",
            "widget_name": "diffusion_model",
        }]
        result = matcher.match(items)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["matched_value"], "mycoolmodel.safetensors")

    def test_single_word_base_precision_variant_reverse(self):
        """反向：请求本体 mycoolmodel，本地有带精度后缀的同名文件也应匹配。"""
        models = [
            {"filename": "mycoolmodel_fp16.safetensors",
             "path": "D:/models/mycoolmodel_fp16.safetensors",
             "type": "checkpoints"},
        ]
        matcher = self._make_matcher(models)
        items = [{
            "id": 1,
            "current": "mycoolmodel.safetensors",
            "node_type": "ModelLoader",
            "widget_name": "diffusion_model",
        }]
        result = matcher.match(items)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["matched_value"], "mycoolmodel_fp16.safetensors")


class TestOnlineComponentFilter(unittest.TestCase):
    """在线侧：主模型请求剔除组件候选；组件请求保留组件候选。"""

    def setUp(self):
        self.searcher = ModelSearcher()

    def test_main_model_drops_component_candidates(self):
        candidates = [
            {"name": "minimax_music3_text_encoder_pruned_int8_convrot.safetensors",
             "filename": "minimax_music3_text_encoder_pruned_int8_convrot.safetensors",
             "source": "HuggingFace", "score": 0.98},
            {"name": "minimax_music3_dav.safetensors",
             "filename": "minimax_music3_dav.safetensors",
             "source": "ModelScope", "score": 0.95},
            {"name": "minimax_music3_dit_fp16.safetensors",
             "filename": "minimax_music3_dit_fp16.safetensors",
             "source": "HuggingFace", "score": 0.97},
        ]
        filtered = self.searcher._filter_component_candidates(
            "minimax_music3_dit_fp16.safetensors", candidates)
        names = [c["name"] for c in filtered]
        self.assertNotIn(
            "minimax_music3_text_encoder_pruned_int8_convrot.safetensors", names,
            "主模型请求不应返回 text_encoder 组件")
        self.assertNotIn(
            "minimax_music3_dav.safetensors", names,
            "主模型请求不应返回 dav 组件")
        self.assertIn("minimax_music3_dit_fp16.safetensors", names,
                      "主模型本体候选应保留")

    def test_component_request_keeps_component_candidates(self):
        candidates = [
            {"name": "sd_xl_base_1.0_vae.safetensors",
             "filename": "sd_xl_base_1.0_vae.safetensors",
             "source": "HuggingFace", "score": 0.90},
            {"name": "sd_xl_base_1.0.safetensors",
             "filename": "sd_xl_base_1.0.safetensors",
             "source": "HuggingFace", "score": 0.90},
        ]
        filtered = self.searcher._filter_component_candidates(
            "sd_xl_base_1.0_vae.safetensors", candidates)
        names = [c["name"] for c in filtered]
        self.assertIn("sd_xl_base_1.0_vae.safetensors", names,
                      "组件请求应保留 vae 组件候选")


class TestOnlineSourcePreference(unittest.TestCase):
    """在线侧：相近分数下官方国内镜像（HF/ModelScope）应胜出。"""

    def setUp(self):
        self.searcher = ModelSearcher()
        # 显式注入源优先级权重，使本用例不依赖 envs/config.json 是否存在，
        # 纯粹验证「相近分数下官方国内镜像（HF/ModelScope）应胜出」的加权逻辑。
        self.searcher.config.setdefault("searcher", {})["source_preference"] = {
            "HuggingFace": 1.0,
            "ModelScope": 1.0,
            "CNB": 0.95,
            "Civitai": 0.9,
            "Liblib": 0.9,
        }

    def test_hf_beats_liblib_on_tie(self):
        candidates = [
            {"name": "some_main.safetensors", "source": "Liblib", "score": 0.70},
            {"name": "some_main.safetensors", "source": "HuggingFace", "score": 0.70},
        ]
        weighted = self.searcher._apply_source_preference(list(candidates))
        weighted.sort(key=lambda x: x["score"], reverse=True)
        self.assertEqual(weighted[0]["source"], "HuggingFace")

    def test_modelscope_beats_civitai_on_tie(self):
        candidates = [
            {"name": "some_main.safetensors", "source": "Civitai", "score": 0.70},
            {"name": "some_main.safetensors", "source": "ModelScope", "score": 0.70},
        ]
        weighted = self.searcher._apply_source_preference(list(candidates))
        weighted.sort(key=lambda x: x["score"], reverse=True)
        self.assertEqual(weighted[0]["source"], "ModelScope")


if __name__ == "__main__":
    unittest.main()
