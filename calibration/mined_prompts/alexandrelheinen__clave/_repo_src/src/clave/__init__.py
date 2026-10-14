"""CLAVE learning platform.

Corpus manifests resolved by digest, run records that make a result traceable
to its inputs, a single seeding entry point, and the research document checker
deferred from v0.1.0.

Nothing in this package imports a deep learning framework. PyTorch is declared
as an optional extra and is taken up at v0.7.0.
"""

from clave.errors import ClaveError

__all__ = ["ClaveError"]
