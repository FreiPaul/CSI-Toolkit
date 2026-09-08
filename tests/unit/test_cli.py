"""The CLI has to report failure, whichever way it is started."""

import shutil
import subprocess
import sys
from pathlib import Path

from csi_toolkit.main import main


def test_main_reports_a_missing_input_file(tmp_path):
    assert main(["process", str(tmp_path / "absent.csv"), str(tmp_path / "out.csv")]) != 0


def test_main_succeeds_without_touching_sys_argv(capsys):
    assert main(["process", "--list-features"]) == 0
    assert "mean_amp" in capsys.readouterr().out


def _run(command, tmp_path):
    return subprocess.run(
        command + ["process", str(tmp_path / "absent.csv"), str(tmp_path / "out.csv")],
        capture_output=True,
        text=True,
        timeout=60,
    ).returncode


def test_running_as_a_module_propagates_the_exit_code(tmp_path):
    assert _run([sys.executable, "-m", "csi_toolkit"], tmp_path) != 0


def test_both_entry_points_agree_on_the_exit_code(tmp_path):
    beside_interpreter = Path(sys.executable).with_name("csi-toolkit")
    console_script = str(beside_interpreter) if beside_interpreter.exists() else shutil.which("csi-toolkit")
    assert console_script, "csi-toolkit is not installed in this environment"
    from_module = _run([sys.executable, "-m", "csi_toolkit"], tmp_path)
    from_script = _run([console_script], tmp_path)
    assert from_script != 0
    assert from_module == from_script
