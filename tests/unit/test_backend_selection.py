"""Choosing a backend for the live window must not fail quietly."""

import pytest

from csi_toolkit.visualization import live_plotter


@pytest.fixture
def refusals(monkeypatch):
    """Record which backends were asked for, refusing every one of them."""
    asked = []

    def refuse(name):
        asked.append(name)
        raise ImportError(f"no module for {name}")

    monkeypatch.delenv("MPLBACKEND", raising=False)
    monkeypatch.setattr(live_plotter.matplotlib, "use", refuse)
    return asked


def test_a_configured_backend_is_left_alone(monkeypatch):
    monkeypatch.setenv("MPLBACKEND", "Agg")
    monkeypatch.setattr(live_plotter.matplotlib, "use", lambda name: pytest.fail("switched away"))
    live_plotter._use_interactive_backend()


def test_the_first_usable_backend_wins(monkeypatch):
    monkeypatch.delenv("MPLBACKEND", raising=False)
    used = []
    monkeypatch.setattr(live_plotter.matplotlib, "use", used.append)
    live_plotter._use_interactive_backend()
    assert len(used) == 1


def test_no_usable_backend_is_reported(refusals):
    with pytest.raises(RuntimeError, match="MPLBACKEND"):
        live_plotter._use_interactive_backend()
    assert "TkAgg" in refusals


def test_macos_keeps_the_cross_platform_fallbacks(refusals, monkeypatch):
    monkeypatch.setattr(live_plotter.platform, "system", lambda: "Darwin")
    with pytest.raises(RuntimeError):
        live_plotter._use_interactive_backend()
    assert refusals == ["MacOSX", "TkAgg", "Qt5Agg"]
