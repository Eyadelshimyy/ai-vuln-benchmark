"""Column-level schema inference for tabular files.

Complements ``ingest/detect.py`` (which classifies file/document type and
sheet structure) with per-column type, nullability, and uniqueness profiling.
"""

from womblex.profile.columns import (
    ColumnProfile,
    TableProfile,
    profile_dataframe,
    profile_file,
)

__all__ = [
    "ColumnProfile",
    "TableProfile",
    "profile_dataframe",
    "profile_file",
]
