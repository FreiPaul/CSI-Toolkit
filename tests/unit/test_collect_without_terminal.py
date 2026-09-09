"""Collection has to run where no terminal is attached: docker, cron, a pipe."""

import contextlib
import subprocess
import sys

from csi_toolkit.collection import CollectorConfig, SerialCollector


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


def test_collect_survives_a_missing_stdin(tmp_path, monkeypatch):
    monkeypatch.setattr(sys, "stdin", None)
    config = CollectorConfig(
        serial_port=str(tmp_path / "absent-device"),
        baudrate=921600,
        flush_interval=1,
        output_dir=str(tmp_path / "out"),
    )
    with contextlib.suppress(ConnectionError):
        SerialCollector(config).start()
