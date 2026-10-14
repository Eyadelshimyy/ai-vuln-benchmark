"""
Shared utility functions.
"""
import math
import re
from typing import Any


def safe(val: Any, decimals: int = 4) -> float | None:
    """Convert numpy/pandas numerics to JSON-safe Python floats."""
    if val is None:
        return None
    f = float(val)
    return None if (math.isnan(f) or math.isinf(f)) else round(f, decimals)


def company_search_name(full_name: str) -> str:
    """Extract a search-friendly company name from a legal name.

    Strips legal suffixes so searches use the brand name people actually write
    in articles and social posts rather than the full registered name.

    Examples:
        "FormFactor, Inc."   -> "FormFactor"
        "Tesla, Inc."        -> "Tesla"
        "NVIDIA Corporation" -> "NVIDIA"
        "Alphabet Inc."      -> "Alphabet"
    """
    if not full_name:
        return ""
    name = full_name.split(",")[0]
    name = re.sub(r'\s+(Inc\.?|Corp\.?|Corporation|Ltd\.?|LLC|Co\.)$', '', name, flags=re.IGNORECASE)
    return name.strip()
