"""Slice — human-structure understanding engine.

Converts a person image / illustration into a stick-figure skeleton,
predicts the full body from statistical anatomy priors, and stores the
result as reusable 2D/3D knowledge (Knowledge JSON).

All results are *inferences*: every joint carries a confidence score and
an observed/predicted state. A single image can never prove what is under
the clothes or off-screen; Slice records what it saw and what it guessed
as separate facts.
"""

__version__ = "0.1.0"
