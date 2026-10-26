"""ldm: selective legacy data migration job (Db2 fixed-width unload -> Azure SQL)."""

from decimal import DefaultContext, getcontext

__version__ = "0.1.0"

# DECIMAL(31,8) values and their exact sums need more than Python's default 28 digits of precision.
DECIMAL_PRECISION = 64
DefaultContext.prec = DECIMAL_PRECISION
getcontext().prec = DECIMAL_PRECISION
