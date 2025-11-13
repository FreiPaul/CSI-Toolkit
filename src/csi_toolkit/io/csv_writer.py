"""CSV writing utilities with flush control."""

import csv
import os
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Dict, Any

from ..core.constants import (
    CSV_HEADER,
    CSV_TEMP_FILENAME,
    CSV_FILE_PREFIX,
    CSV_DATE_FORMAT,
)


class CSVWriter:
    """CSV writer with flush control and automatic file renaming."""

    def __init__(
        self,
        output_dir: str = "data",
        flush_interval: int = 1,
        temp_filename: str = CSV_TEMP_FILENAME,
        header: List[str] = None,
    ):
        """
        Initialize CSV writer.

        Args:
            output_dir: Directory for output files
            flush_interval: Number of rows between flushes
            temp_filename: Temporary filename during collection
            header: CSV header (defaults to CSV_HEADER)
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.flush_interval = flush_interval
        self.temp_filename = temp_filename
        self.header = header or CSV_HEADER

        self.file_path = self.output_dir / self.temp_filename
        self.file = None
        self.writer = None
        self.row_count = 0
        self.start_time = None

    def __enter__(self):
        """Context manager entry."""
        self.open()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()

    def open(self):
        """Open CSV file and write header."""
        self.start_time = datetime.now()
        self.file = open(self.file_path, 'w', newline='')
        self.writer = csv.writer(self.file)
        self.writer.writerow(self.header)
        self.file.flush()
        self.row_count = 0

    def write_row(self, row: List[Any]):
        """
        Write a row to the CSV file.

        Args:
            row: List of values to write
        """
        if not self.writer:
            raise RuntimeError("CSV file not open. Call open() first.")

        self.writer.writerow(row)
        self.row_count += 1

        # Flush based on interval
        if self.row_count % self.flush_interval == 0:
            self.flush()

    def write_dict(self, row_dict: Dict[str, Any]):
        """
        Write a dictionary row to the CSV file.

        Args:
            row_dict: Dictionary with keys matching header columns
        """
        row = [row_dict.get(col, '') for col in self.header]
        self.write_row(row)

    def flush(self):
        """Flush the file to disk."""
        if self.file:
            self.file.flush()
            os.fsync(self.file.fileno())

    def close(self):
        """Close the file and rename to timestamped filename."""
        if self.file:
            self.flush()
            self.file.close()
            self.file = None
            self.writer = None

            # Rename to timestamped file
            if self.start_time and self.file_path.exists():
                timestamp = self.start_time.strftime(CSV_DATE_FORMAT)
                new_filename = f"{CSV_FILE_PREFIX}_{timestamp}.csv"
                new_path = self.output_dir / new_filename

                try:
                    # If destination already exists, add a counter
                    if new_path.exists():
                        counter = 1
                        while True:
                            new_filename = f"{CSV_FILE_PREFIX}_{timestamp}_{counter}.csv"
                            new_path = self.output_dir / new_filename
                            if not new_path.exists():
                                break
                            counter += 1

                    self.file_path.rename(new_path)
                    print(f"Data saved to: {new_path}")
                    return new_path
                except Exception as e:
                    print(f"Warning: Could not rename file: {e}")
                    print(f"File remains at: {self.file_path}")

        return self.file_path

    def get_row_count(self) -> int:
        """Get the number of rows written."""
        return self.row_count