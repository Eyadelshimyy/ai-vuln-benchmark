"""Putting CLAVE's decision on ROS 2, which is the middleware FRET runs on.

Nothing here plans, moves or grasps. FRET already has a pick-and-place state
machine, a joint-space controller and ARCO planning behind them, so this package
carries a message and stops.

The modules split on whether they need a ROS installation. [decisions] decodes
the published contract and needs nothing but `cbor2`, so it runs wherever the
gate runs. [publisher] imports ROS lazily and says so when it is absent.
"""
