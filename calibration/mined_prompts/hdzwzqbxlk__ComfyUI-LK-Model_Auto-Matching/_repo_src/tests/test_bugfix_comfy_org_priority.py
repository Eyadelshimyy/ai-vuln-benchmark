# -*- coding: utf-8 -*-
"""
回归测试：在线搜索优先返回 Comfy-Org / unsloth 官方镜像。

对应用户反馈： comfy 官方模型（Comfy-Org）和 unsloth 镜像在搜索结果中不出现，
反而返回原始仓库（MiniMaxAI/MiniMax-Music3）或第三方非官方源（dummy9996）。
"""
import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from core.searcher import HuggingFaceFileSearchProvider, ModelScopeFileSearchProvider, ModelSearcher


class TestNamespacePreference:
    """验证命名空间加权让 Comfy-Org / unsloth 优先。"""

    def test_comfy_org_boosted_over_original(self):
        ms = ModelSearcher()
        candidates = [
            {"name": "MiniMaxAI/MiniMax-Music3", "source": "HuggingFace (Exact File)", "score": 0.95},
            {"name": "Comfy-Org/MiniMax-Music-3", "source": "HuggingFace (Exact File)", "score": 0.95},
        ]
        result = ms._apply_namespace_preference(candidates)
        scores = {c["name"]: c["score"] for c in result}
        assert scores["Comfy-Org/MiniMax-Music-3"] > scores["MiniMaxAI/MiniMax-Music3"]
        assert scores["Comfy-Org/MiniMax-Music-3"] == pytest.approx(0.95 * 1.15, rel=1e-3)

    def test_unsloth_boosted(self):
        ms = ModelSearcher()
        candidates = [
            {"name": "MiniMaxAI/MiniMax-H3", "source": "HuggingFace (Exact File)", "score": 0.92},
            {"name": "unsloth/MiniMax-H3", "source": "HuggingFace (Exact File)", "score": 0.92},
        ]
        result = ms._apply_namespace_preference(candidates)
        scores = {c["name"]: c["score"] for c in result}
        assert scores["unsloth/MiniMax-H3"] > scores["MiniMaxAI/MiniMax-H3"]
        assert scores["unsloth/MiniMax-H3"] == pytest.approx(0.92 * 1.08, rel=1e-3)

    def test_namespace_preference_case_insensitive(self):
        ms = ModelSearcher()
        candidates = [
            {"name": "comfy-org/MiniMax-Music-3", "source": "HuggingFace (Exact File)", "score": 0.95},
        ]
        result = ms._apply_namespace_preference(candidates)
        assert result[0]["score"] == pytest.approx(0.95 * 1.15, rel=1e-3)


class TestHuggingFacePriorityAuthors:
    """验证 HF 优先搜索 Comfy-Org / unsloth 命名空间。"""

    def test_default_priority_authors(self):
        provider = HuggingFaceFileSearchProvider({})
        authors = provider._get_priority_authors()
        assert "Comfy-Org" in authors
        assert "unsloth" in authors

    def test_config_override_priority_authors(self):
        cfg = {"searcher": {"api": {"huggingface": {"priority_authors": ["Foo"]}}}}
        provider = HuggingFaceFileSearchProvider(cfg)
        assert provider._get_priority_authors() == ["Foo"]


class TestFocusedSearchQuery:
    """核心修复：优先级检索用聚焦的 top-2 查询，避免 'minimax music dit' 落空。"""

    def test_focused_top2_drops_type_token(self):
        p = HuggingFaceFileSearchProvider({})
        # minimax_music3_dit -> keywords [minimax, music, dit]; 聚焦 top2 应丢弃 'dit'
        kw = p._build_search_queries(["minimax", "music", "dit"], 2)
        assert kw == ["minimax", "music"]

    def test_focused_drops_pure_digit(self):
        p = HuggingFaceFileSearchProvider({})
        kw = p._build_search_queries(["minimax", "music", "3"], 2)
        assert kw == ["minimax", "music"]

    def test_focused_single_keyword_fallback(self):
        p = HuggingFaceFileSearchProvider({})
        kw = p._build_search_queries(["minimax"], 2)
        assert kw == ["minimax"]


class TestModelScopePriorityOrgs:
    """验证 ModelScope 走统一的 priority_organizations（组织优先），而非硬编码 per-model 仓库。"""

    def test_default_priority_orgs(self):
        p = ModelScopeFileSearchProvider({})
        orgs = p._get_priority_orgs()
        assert "Comfy-Org" in orgs
        assert "unsloth" in orgs

    def test_config_override_priority_orgs(self):
        cfg = {"searcher": {"priority_organizations": {"modelscope": ["MyOrg"]}}}
        p = ModelScopeFileSearchProvider(cfg)
        assert p._get_priority_orgs() == ["MyOrg"]

    def test_no_hardcoded_minimax_repos(self):
        # 用户明确 MiniMax 是"其他组织"放出，不应再硬编码到 PRIORITY_REPOS
        assert "minimax" not in ModelScopeFileSearchProvider.PRIORITY_REPOS
        assert "music3" not in ModelScopeFileSearchProvider.PRIORITY_REPOS
        assert "h3" not in ModelScopeFileSearchProvider.PRIORITY_REPOS


class TestNamespacePreferenceFromPriorityOrgs:
    """priority_organizations 中的组织即使未列于 namespace_preference 也应被加权。"""

    def test_priority_org_default_boost(self):
        cfg = {
            "searcher": {
                "priority_organizations": {"huggingface": ["Comfy-Org", "unsloth", "NewOrg"]},
                "namespace_preference": {"Comfy-Org": 1.15, "unsloth": 1.08},
            }
        }
        ms = ModelSearcher()
        ms.config = cfg
        candidates = [
            {"name": "NewOrg/Foo", "source": "HuggingFace (Exact File)", "score": 0.90},
            {"name": "SomeOther/Foo", "source": "HuggingFace (Exact File)", "score": 0.90},
        ]
        result = ms._apply_namespace_preference(candidates)
        scores = {c["name"]: c["score"] for c in result}
        # NewOrg 来自 priority_organizations，应获得默认 1.12 加权
        assert scores["NewOrg/Foo"] == pytest.approx(0.90 * 1.12, rel=1e-3)
        assert scores["NewOrg/Foo"] > scores["SomeOther/Foo"]
