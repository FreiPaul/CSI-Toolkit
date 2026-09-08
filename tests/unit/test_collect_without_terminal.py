"""Collection has to run where no terminal is attached: docker, cron, a pipe."""

import subprocess
import sys


def _collect_without_stdin(tmp_path):
    return subprocess.run(
        [
            sys.executable, "-m", "csi_toolkit", "collect",
            "--port", str(tmp_path / "absent-device"),
            "--output-dir", str(tmp_path / "out"),
        ],
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        timeout=60,
    )


def test_collect_does_not_touch_the_terminal_when_there_is_none(tmp_path):
    result = _collect_without_stdin(tmp_path)
    assert "termios" not in result.stderr
    assert "Exception in thread" not in result.stderr


def test_collect_reports_a_missing_device(tmp_path):
    assert _collect_without_stdin(tmp_path).returncode != 0


def test_collect_says_that_labeling_is_unavailable(tmp_path):
    assert "No terminal attached" in _collect_without_stdin(tmp_path).stdout
