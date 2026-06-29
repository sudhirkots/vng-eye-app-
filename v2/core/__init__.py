"""V2 prototype core modules.

V2 is a clean re-derivation of the iris-tracking pipeline from the clinical reasoning
model (find face -> find eye opening -> find iris -> measure iris in eye-local
coordinates). It is intentionally independent of the V1 codebase: nothing in v2/ may
import from src/core, iris_tracker.py, or any V1 module.
"""
