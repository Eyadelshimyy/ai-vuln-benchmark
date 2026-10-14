"""Base exception for the package.

The house Python standards require one exception per package so a caller can
catch the package rather than each individual error type.
"""


class ClaveError(Exception):
    """Base class for every error this package raises."""


class ManifestError(ClaveError):
    """The corpus manifest is malformed, or names an artifact twice."""


class UnknownArtifactError(ClaveError):
    """A run referenced an artifact that the manifest does not describe."""
