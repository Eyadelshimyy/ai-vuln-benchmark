"""镜像感知的 URL 重写（确定性、国内优先、零网络）。

统一把 HuggingFace / ModelScope 的下载 URL 路由到用户配置的镜像端点
（如 hf-mirror.com、modelscope.ai），而匹配/解析逻辑完全不受影响。

设计原则（对齐 docs/MODEL_MATCHING_FEASIBILITY.md §4.2）：
- 端点未配置时，所有函数原样透传 -> 对未配置镜像的用户行为零改变；
- 只替换 host，query/path 原样保留 -> 确定性 URL 不变形；
- 非目标域（如 civitai.com）一律不处理 -> Civitai 下载无国内镜像，保持原样。
"""
import urllib.parse

try:
    from .config import get_all_config
except ImportError:  # 允许独立 import（测试 / 脚本场景）
    from config import get_all_config


def _hf_endpoint():
    try:
        return (get_all_config().get("mirrors", {}) or {}).get("hf_endpoint", "") or ""
    except Exception:
        return ""


def _modelscope_endpoint():
    try:
        return (get_all_config().get("mirrors", {}) or {}).get("modelscope_endpoint", "") or ""
    except Exception:
        return ""


def _replace_host(url, endpoint):
    """仅替换 host（scheme 取端点或默认 https），其余部分保留。"""
    endpoint = (endpoint or "").strip().rstrip("/")
    if not endpoint:
        return url
    try:
        parsed = urllib.parse.urlparse(url)
        if not parsed.netloc:
            return url
        new = urllib.parse.urlparse(endpoint)
        rewritten = parsed._replace(scheme=new.scheme or "https", netloc=new.netloc)
        return urllib.parse.urlunparse(rewritten)
    except Exception:
        return url


def rewrite_hf_url(url):
    """把 huggingface.co URL 重写到配置的 HF_ENDPOINT 镜像。

    非 huggingface.co 域名或端点未配置时原样返回。
    """
    endpoint = _hf_endpoint().strip()
    if not endpoint:
        return url
    try:
        if urllib.parse.urlparse(url).netloc.lower() != "huggingface.co":
            return url
    except Exception:
        return url
    return _replace_host(url, endpoint)


def rewrite_modelscope_url(url):
    """把 modelscope.cn / modelscope.ai URL 重写到配置的 MODELSCOPE_ENDPOINT 镜像。"""
    endpoint = _modelscope_endpoint().strip()
    if not endpoint:
        return url
    try:
        if urllib.parse.urlparse(url).netloc.lower() not in ("modelscope.cn", "modelscope.ai"):
            return url
    except Exception:
        return url
    return _replace_host(url, endpoint)
