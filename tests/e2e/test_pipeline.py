"""End-to-end tests for the processing pipeline CLI."""

import csv
import os
import tempfile
import unittest
from csi_toolkit.core.constants import CSV_HEADER
from csi_toolkit.processing.feature_extractor import FeatureExtractor


class TestEndToEndPipeline(unittest.TestCase):
    """Test full processing pipeline end-to-end."""

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.input_csv = os.path.join(self.temp_dir.name, "input.csv")
        self.output_csv = os.path.join(self.temp_dir.name, "output.csv")

        # Create sample raw CSV file with 400 packets (yielding 4 windows of size 100)
        with open(self.input_csv, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(CSV_HEADER)
            for i in range(400):
                row = [
                    "CSI_DATA",
                    str(i),
                    "1a:00:00:00:00:00",
                    "-60",
                    "11",
                    "-93",
                    "12",
                    "53",
                    "11",
                    "2026-09-08 12:00:00.000",
                    "128",
                    "0",
                    "384",
                    "0",
                    "[10, 20, 30, 40]",
                    str([15.0] * 64),
                    "0",
                ]
                writer.writerow(row)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_feature_extraction_pipeline(self):
        extractor = FeatureExtractor()
        results = extractor.process_file(
            input_csv=self.input_csv,
            output_csv=self.output_csv,
            window_size=100,
        )
        self.assertGreater(len(results), 0)
        self.assertTrue(os.path.exists(self.output_csv))

        with open(self.output_csv, "r") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
            self.assertGreater(len(rows), 0)
            self.assertIn("window_id", rows[0])


if __name__ == "__main__":
    unittest.main()
