"""The software-in-the-loop runtime: frame in, published decision out.

Python steps the world and runs inference. Rust checks what inference proposed
against a workspace envelope and publishes what survives. The two are separate
processes on purpose, so the layer that can override a model links no part of
one.
"""
