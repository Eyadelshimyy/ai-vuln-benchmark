# comfy_test_bootstrap.py — pytest 引导插件（通过 pytest.ini 的 addopts = -p comfy_test_bootstrap 加载）
#
# 根目录名含连字符（ComfyUI-LK-Model_Auto-Matching），不是合法 Python 标识符。
# pytest 的 resolve_package_path 遇到非标识符目录名会返回 None，进而把根 __init__.py
# 当作顶层模块 "__init__" 直接加载——而该文件是 ComfyUI 节点入口，含 `import server`
# 与相对导入 `from .core...`，在测试环境下必然导入失败，导致全部用例收集失败。
#
# 解决：在 sys.modules 注册名为 "__init__" 的占位模块（__file__ 指向根 __init__.py），
# 使 pytest 的 import_module("__init__") 命中缓存、跳过对根 __init__.py 的实际加载。
# 该插件作为顶层模块被导入（cwd 在 sys.path 中），自身不会触发根包导入。
import os
import sys
import types
from unittest.mock import MagicMock

ROOT = os.path.dirname(os.path.abspath(__file__))
PKG_NAME = os.path.basename(ROOT)

_placeholder = types.ModuleType("__init__")
_placeholder.__path__ = [ROOT]
_placeholder.__file__ = os.path.join(ROOT, "__init__.py")
_placeholder.__package__ = PKG_NAME
sys.modules.setdefault("__init__", _placeholder)
sys.modules.setdefault(PKG_NAME, _placeholder)

# Stub ComfyUI 私有依赖，避免任何 core 模块顶层 `import folder_paths` 失败
if "folder_paths" not in sys.modules:
    sys.modules["folder_paths"] = MagicMock()
