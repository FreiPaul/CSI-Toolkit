"""Importing the package must leave the caller's matplotlib backend alone."""

import subprocess
import sys

PROBE = (
    "import matplotlib;"
    "before = matplotlib.get_backend();"
    "import csi_toolkit.visualization;"
    "print(before, matplotlib.get_backend())"
)


def _backends(env_backend):
    import os

    env = dict(os.environ)
    if env_backend is None:
        env.pop("MPLBACKEND", None)
    else:
        env["MPLBACKEND"] = env_backend
    result = subprocess.run(
        [sys.executable, "-c", PROBE], capture_output=True, text=True, timeout=60, env=env
    )
    assert result.returncode == 0, result.stderr
    return result.stdout.split()


def test_importing_visualization_keeps_the_configured_backend():
    before, after = _backends("Agg")
    assert before.lower() == "agg"
    assert after == before


def test_importing_visualization_keeps_the_default_backend():
    before, after = _backends(None)
    assert after == before
