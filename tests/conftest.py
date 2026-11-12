"""Test-wide settings: the HTTP politeness layer must not slow the unit tests down.  Real rates are verified in
test_politeness.py with an injected clock."""
import os

os.environ.setdefault("PWPF_HTTP_RATE_SCALE", "1000")
