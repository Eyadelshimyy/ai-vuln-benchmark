# 根级 conftest：根 __init__.py（ComfyUI 节点入口）不是测试用例，避免被收集。
# 根包的占位注册与 ComfyUI 依赖 stub 已由 comfy_test_bootstrap 插件（pytest.ini addopts）完成。
import sys

import pytest

collect_ignore = ["__init__.py"]


@pytest.fixture(autouse=True)
def _isolate_external_db(monkeypatch):
    """隔离 matcher 单测与外部 SQLite 数据库。

    ``ModelMatcher.match()`` 默认走 ``use_db_fallback``，会查询随仓库发布的
    ``core/data/models.db``（external_models 表）。该库含真实远程模型条目，
    会与测试里 mock 的 scanner 本地模型混淆，导致“不应匹配”的用例误判为匹配、
    甚至把请求文件名自身当成本地匹配返回。

    单测只应基于 mock 的 scanner 本地模型做断言，故在此 stub 掉
    ``db.lookup_modelsdb``，使其恒返回 (None, 0)。

    注意：``core.database`` 由 ``match()`` 内部惰性导入，fixture 建立时它可能尚未
    进入 sys.modules，因此这里显式 import 以确保补丁生效。
    """
    try:
        import core.database as db_mod
    except Exception:
        yield
        return
    if getattr(db_mod, "db", None) is not None:
        monkeypatch.setattr(
            db_mod.db,
            "lookup_modelsdb",
            lambda filename, expected_types=None: (None, 0),
        )
    yield
