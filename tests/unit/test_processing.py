"""Unit tests for windowing and feature extraction."""

import unittest
from csi_toolkit.processing.windowing import CSISample


class TestProcessingWindowing(unittest.TestCase):
    """Test CSISample construction."""

    def test_csisample_from_csv_row(self):
        row = {
            "seq": "100",
            "local_timestamp": "2026-09-08 12:00:00.123",
            "mac": "1a:00:00:00:00:00",
            "amplitudes": "[1.0, 2.0, 3.0]",
            "label": "1",
        }
        sample = CSISample.from_csv_row(row, labeled_mode=True)
        self.assertEqual(sample.seq, 100)
        self.assertEqual(sample.timestamp, "2026-09-08 12:00:00.123")
        self.assertEqual(sample.mac, "1a:00:00:00:00:00")
        self.assertEqual(sample.label, 1)
        self.assertEqual(sample.amplitudes, [1.0, 2.0, 3.0])


if __name__ == "__main__":
    unittest.main()
