"""Make ``src/`` importable so tests can ``import hhl_algorithm`` directly."""

import os
import sys

SRC = os.path.join(os.path.dirname(os.path.dirname(__file__)), "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)
