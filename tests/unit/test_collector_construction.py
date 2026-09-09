"""A collector owns the shutdown signals only while it is collecting."""

import contextlib
import signal
import threading

from csi_toolkit.collection import CollectorConfig, SerialCollector


def _config(tmp_path):
    return CollectorConfig(
        serial_port=str(tmp_path / "absent-device"),
        baudrate=921600,
        flush_interval=1,
        output_dir=str(tmp_path / "out"),
    )


def test_constructing_a_collector_leaves_the_signal_handlers_alone(tmp_path):
    before = signal.getsignal(signal.SIGTERM)
    SerialCollector(_config(tmp_path))
    assert signal.getsignal(signal.SIGTERM) is before


def test_a_collector_can_be_constructed_off_the_main_thread(tmp_path):
    failure = []

    def build():
        try:
            SerialCollector(_config(tmp_path))
        except Exception as exc:  # noqa: BLE001 - the test reports whatever went wrong
            failure.append(exc)

    worker = threading.Thread(target=build)
    worker.start()
    worker.join(timeout=30)
    assert not failure, failure


def test_the_shutdown_signals_are_taken_over_and_handed_back(tmp_path):
    collector = SerialCollector(_config(tmp_path))
    before = signal.getsignal(signal.SIGTERM)
    try:
        collector._install_signal_handlers()
        assert signal.getsignal(signal.SIGTERM) is not before
        collector._restore_signal_handlers()
        assert signal.getsignal(signal.SIGTERM) is before
    finally:
        signal.signal(signal.SIGTERM, before)


def test_a_finished_collection_leaves_the_signals_as_it_found_them(tmp_path):
    collector = SerialCollector(_config(tmp_path))
    before = signal.getsignal(signal.SIGTERM)
    with contextlib.suppress(Exception):
        collector.start()  # the device is absent, so the run ends immediately
    assert signal.getsignal(signal.SIGTERM) is before
