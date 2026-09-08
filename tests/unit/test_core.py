"""Unit tests for core parser, constants, and data structures."""

import unittest
from csi_toolkit.core.constants import CSV_HEADER
from csi_toolkit.core.parser import parse_csi_line, parse_amplitude_json


class TestCoreParser(unittest.TestCase):
    """Test core parser functions."""

    def test_parse_csi_line_valid(self):
        line = 'CSI_DATA,32633015,1a:00:00:00:00:00,-60,11,-93,12,53,11,2054426858,128,0,384,0,"[0,0,0,0,15,46]"'
        fields = parse_csi_line(line)
        self.assertIsNotNone(fields)
        self.assertEqual(len(fields), 14)
        self.assertEqual(fields[0], "32633015")
        self.assertEqual(fields[1], "1a:00:00:00:00:00")
        self.assertEqual(fields[8], "2054426858")  # device_timestamp
        self.assertEqual(fields[9], "128")  # sig_len

    def test_parse_csi_line_invalid_prefix(self):
        line = "LOG_DATA,123,456"
        fields = parse_csi_line(line)
        self.assertIsNone(fields)

    def test_parse_amplitude_json(self):
        json_str = "[10, 20, 30, 40]"
        vals = parse_amplitude_json(json_str)
        self.assertEqual(vals, [10.0, 20.0, 30.0, 40.0])

    def test_csv_header_schema(self):
        self.assertIn("type", CSV_HEADER)
        self.assertIn("seq", CSV_HEADER)
        self.assertIn("mac", CSV_HEADER)
        self.assertIn("local_timestamp", CSV_HEADER)
        self.assertIn("sig_len", CSV_HEADER)
        self.assertIn("amplitudes", CSV_HEADER)
        self.assertIn("label", CSV_HEADER)


if __name__ == "__main__":
    unittest.main()
