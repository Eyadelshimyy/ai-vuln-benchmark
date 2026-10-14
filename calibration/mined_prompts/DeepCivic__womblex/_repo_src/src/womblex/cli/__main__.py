"""Enable ``python -m womblex.cli`` to run the CLI.

The console script ``womblex`` (declared in pyproject.toml) is the primary
entry point, but ``cli`` is a package, so ``python -m womblex.cli`` would
otherwise fail with "is a package and cannot be directly executed". This
module routes the ``-m`` invocation to the same ``main()``.
"""
from womblex.cli import main

raise SystemExit(main())
