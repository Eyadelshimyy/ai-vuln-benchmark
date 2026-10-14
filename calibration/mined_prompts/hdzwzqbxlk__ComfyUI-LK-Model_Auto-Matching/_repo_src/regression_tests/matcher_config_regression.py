import json
import os
import sys
import tempfile
import types
import unittest

ROOT = os.path.dirname(os.path.dirname(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

folder_paths_stub = types.ModuleType("folder_paths")
folder_paths_stub.get_folder_paths = lambda *args, **kwargs: []
sys.modules.setdefault("folder_paths", folder_paths_stub)

import core.database as database_module
import core.matcher as matcher_module
from core.config import get_matcher_config, override_matcher_config, reset_matcher_config


class MatcherConfigTests(unittest.TestCase):
    def setUp(self):
        reset_matcher_config()

    def tearDown(self):
        reset_matcher_config()

    def test_db_lookup_uses_configured_semantic_threshold(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "models.db")
            db = database_module.ModelDatabase(db_path)

            payload = {
                "MODELS_DB": {
                    "flux-dev.safetensors": {
                        "repo_id": "example/repo",
                        "path": "flux-dev.safetensors",
                        "filename": "flux-dev.safetensors",
                        "source": "example",
                    }
                }
            }
            json_path = os.path.join(tmpdir, "models_db.json")
            with open(json_path, "w", encoding="utf-8") as handle:
                json.dump(payload, handle)

            inserted = db.import_models_db_json(json_path)
            self.assertEqual(inserted, 1)

            override_matcher_config({"db": {"semantic_min_score": 0.8}})
            info, score = db.lookup_modelsdb("foo.safetensors")

            self.assertIsNone(info)
            self.assertEqual(score, 0)

    def test_db_lookup_matches_prefix_style_filename(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "models.db")
            db = database_module.ModelDatabase(db_path)

            payload = {
                "MODELS_DB": {
                    "flux1-dev-fp8.safetensors": {
                        "repo_id": "example/repo",
                        "path": "flux1-dev-fp8.safetensors",
                        "filename": "flux1-dev-fp8.safetensors",
                        "source": "example",
                    }
                }
            }
            json_path = os.path.join(tmpdir, "models_db.json")
            with open(json_path, "w", encoding="utf-8") as handle:
                json.dump(payload, handle)

            db.import_models_db_json(json_path)
            info, score = db.lookup_modelsdb("FLUX.1-dev.safetensors")

            self.assertIsNotNone(info)
            self.assertGreaterEqual(score, 0.35)
            self.assertEqual(info["filename"], "flux1-dev-fp8.safetensors")

    def test_matcher_respects_strategy_toggle_from_config(self):
        class DummyScanner:
            def get_all_models(self):
                return [{"filename": "exact-model.safetensors", "path": "exact-model.safetensors", "type": "checkpoint"}]

        override_matcher_config({
            "matching": {
                "use_db_fallback": False,
                "use_exact_match": False,
                "use_fuzzy_match": True,
                "use_variant_match": False,
                "use_legacy_match": False,
            }
        })

        matcher = matcher_module.ModelMatcher(DummyScanner())
        matcher.config = get_matcher_config()
        matcher._find_exact_match = lambda item_ctx, ctx: None
        matcher._find_fuzzy_match = lambda item_ctx: {"filename": "exact-model.safetensors", "path": "exact-model.safetensors", "type": "checkpoint"}
        matcher._find_variant_match = lambda item_ctx: None
        matcher._find_legacy_match = lambda item_ctx, ctx: None

        results = matcher.match([
            {"id": "1", "node_type": "checkpoint", "widget_name": "ckpt_name", "current": "sample.safetensors"}
        ])

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["match_type"], "Fuzzy")
        self.assertEqual(results[0]["matched_value"], "exact-model.safetensors")


if __name__ == "__main__":
    unittest.main()
