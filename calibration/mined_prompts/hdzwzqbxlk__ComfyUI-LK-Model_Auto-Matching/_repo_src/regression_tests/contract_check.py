"""Contract-lock regression test for the 6 aiohttp routes in the plugin root.

Mirrors the stub pattern from tests/test_all_db.py: stub external ComfyUI
modules (``server`` and ``folder_paths``) with MagicMocks BEFORE importing the
plugin module, and append the project root to ``sys.path`` so relative imports
(``from .core...``) resolve.

Run directly:
    python regression_tests/test_contract.py

Asserts the 9 points of docs/FRONTEND_BACKEND_CONTRACT.md §4c.
"""

import asyncio
import importlib.util
import json
import os
import sys
import types
import unittest
from unittest.mock import AsyncMock, MagicMock

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# --- Stub ComfyUI's `server` module -----------------------------------------
# `PromptServer.instance.routes.post/get` are used as decorators at import time.
# We make them identity decorators (return the wrapped function unchanged) so the
# real handler functions stay importable.
server_stub = types.ModuleType("server")
_prompt_server = MagicMock()
_routes = MagicMock()


def _identity_decorator(_path):
    def _wrap(fn):
        return fn
    return _wrap


_routes.post.side_effect = _identity_decorator
_routes.get.side_effect = _identity_decorator
_prompt_server.instance.routes = _routes
server_stub.PromptServer = _prompt_server
sys.modules["server"] = server_stub

# --- Stub ComfyUI's `folder_paths` module -----------------------------------
folder_paths_stub = types.ModuleType("folder_paths")
folder_paths_stub.get_folder_paths = lambda *args, **kwargs: []
folder_paths_stub.get_input_directory = lambda *args, **kwargs: "mock_input_dir"
folder_paths_stub.get_output_directory = lambda *args, **kwargs: "mock_output_dir"
folder_paths_stub.get_temp_directory = lambda *args, **kwargs: "mock_temp_dir"
sys.modules["folder_paths"] = folder_paths_stub

# --- Import the plugin module as a *package* so its relative imports work ---
_spec = importlib.util.spec_from_file_location(
    "auto_matcher_plugin",
    os.path.join(ROOT, "__init__.py"),
    submodule_search_locations=[ROOT],
)
plugin = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = plugin
_spec.loader.exec_module(plugin)


class FakeRequest:
    """Minimal aiohttp request stub: `.json()` is an async method."""

    def __init__(self, payload):
        self._payload = payload

    async def json(self):
        return self._payload


def _body(response):
    return json.loads(response.body.decode("utf-8"))


class ContractTests(unittest.TestCase):
    # 1) /match ---------------------------------------------------------------
    def test_match_success_shape_and_type_passthrough(self):
        plugin.matcher.match = lambda items: [{
            "id": "5",
            "node_type": "CheckpointLoader",
            "widget_name": "ckpt_name",
            "original_value": "v1.5.ckpt",
            "matched_value": "v1.5-pruned.safetensors",
            "match_type": "Exact",
            "type": "checkpoints",
        }]
        payload = {"items": [{
            "id": "5",
            "node_type": "CheckpointLoader",
            "widget_name": "ckpt_name",
            "current": "v1.5.ckpt",
        }]}
        response = asyncio.run(plugin.match_models(FakeRequest(payload)))
        self.assertEqual(response.status, 200)
        body = _body(response)
        self.assertIn("matches", body)
        self.assertIsInstance(body["matches"], list)
        item = body["matches"][0]
        for key in ("id", "node_type", "widget_name", "original", "new_value", "match_type", "type"):
            self.assertIn(key, item)
        self.assertIn("new_value", item)
        self.assertNotIn("matched_value", item)
        self.assertEqual(item["type"], "checkpoints")

    # 9) No-rename regression -------------------------------------------------
    def test_match_item_keys_subset_and_no_matched_value(self):
        plugin.matcher.match = lambda items: [{
            "id": "5",
            "node_type": "CheckpointLoader",
            "widget_name": "ckpt_name",
            "original_value": "v1.5.ckpt",
            "matched_value": "v1.5-pruned.safetensors",
            "match_type": "Exact",
        }]
        payload = {"items": [{
            "id": "5",
            "node_type": "CheckpointLoader",
            "widget_name": "ckpt_name",
            "current": "v1.5.ckpt",
        }]}
        response = asyncio.run(plugin.match_models(FakeRequest(payload)))
        item = _body(response)["matches"][0]
        allowed = {"id", "node_type", "widget_name", "original", "new_value", "match_type", "type"}
        self.assertTrue(set(item.keys()).issubset(allowed))
        self.assertNotIn("matched_value", item)

    # 2) /search local --------------------------------------------------------
    def test_search_local_unindexed(self):
        plugin.scanner.find_local_file = lambda filename: "/models/x/" + filename
        plugin.searcher.search = AsyncMock(return_value=[])
        payload = {"items": [{"current": "model.safetensors", "type": "loras"}]}
        response = asyncio.run(plugin.search_models(FakeRequest(payload)))
        self.assertEqual(response.status, 200)
        downloads = _body(response)["downloads"]
        self.assertEqual(len(downloads), 1)
        result = downloads[0]["result"]
        self.assertEqual(result["source"], "Local Disk (Unindexed)")
        self.assertTrue(result["local_path"])
        self.assertEqual(result["url"], "")
        self.assertEqual(downloads[0]["type"], "loras")

    # 3) /search online -------------------------------------------------------
    def test_search_online_passthrough(self):
        plugin.scanner.find_local_file = lambda filename: None
        fake_result = {
            "source": "HuggingFace (Exact File)",
            "name": "Repo/model",
            "filename": "model.safetensors",
            "url": "https://example.com/model.safetensors",
            "pageUrl": "https://example.com/model",
            "score": 0.92,
        }
        plugin.searcher.search = AsyncMock(return_value=[fake_result])
        payload = {"items": [{"current": "model.safetensors", "type": "checkpoints"}]}
        response = asyncio.run(plugin.search_models(FakeRequest(payload)))
        self.assertEqual(response.status, 200)
        downloads = _body(response)["downloads"]
        self.assertEqual(len(downloads), 1)
        result = downloads[0]["result"]
        for key in ("source", "name", "url", "pageUrl", "score"):
            self.assertIn(key, result)
        self.assertEqual(result["url"], "https://example.com/model.safetensors")
        self.assertEqual(downloads[0]["type"], "checkpoints")

    # 4) /refresh-index -------------------------------------------------------
    def test_refresh_index(self):
        plugin.scanner.scan_incremental = lambda: 123
        response = asyncio.run(plugin.refresh_index(FakeRequest({})))
        self.assertEqual(response.status, 200)
        body = _body(response)
        self.assertEqual(body["status"], "ok")
        self.assertIsInstance(body["count"], int)

    # 5) /save-config ---------------------------------------------------------
    def test_save_config(self):
        plugin.searcher.save_config = MagicMock()
        response = asyncio.run(plugin.save_config(FakeRequest({"civitai_api_key": "k"})))
        self.assertEqual(response.status, 200)
        self.assertEqual(_body(response)["status"], "ok")

    # 6) /validate-config -----------------------------------------------------
    def test_validate_config(self):
        response = asyncio.run(plugin.validate_config(FakeRequest({"civitai_api_key": ""})))
        self.assertEqual(response.status, 200)
        body = _body(response)
        self.assertFalse(body["valid"])

        plugin.searcher.validate_api_key = AsyncMock(return_value=(True, "ok"))
        response = asyncio.run(plugin.validate_config(FakeRequest({"civitai_api_key": "k"})))
        body = _body(response)
        self.assertTrue(body["valid"])
        self.assertIsInstance(body["message"], str)

    # 7) /get-config ----------------------------------------------------------
    def test_get_config_version(self):
        response = asyncio.run(plugin.get_config(FakeRequest({})))
        self.assertEqual(response.status, 200)
        body = _body(response)
        self.assertIsInstance(body["version"], str)
        self.assertEqual(body["version"], plugin.__version__)

    # 8) Error envelope (security: no traceback leaked) -----------------------
    def test_error_envelope_no_traceback_leak(self):
        def _boom(items):
            raise RuntimeError("secret /root/path Traceback detail")
        plugin.matcher.match = _boom
        response = asyncio.run(plugin.match_models(FakeRequest({"items": []})))
        self.assertEqual(response.status, 500)
        body = _body(response)
        self.assertIn("error", body)
        self.assertIsInstance(body["error"], str)
        self.assertNotIn("Traceback", body["error"])
        self.assertNotIn("secret", body["error"])
        self.assertNotIn("detail", body)  # raw exception must NOT be in client JSON
        codes = {v for v in vars(plugin.ErrorCode).values() if isinstance(v, str)}
        self.assertIn(body["code"], codes)


if __name__ == "__main__":
    unittest.main()
