"""CSV reading and tailing utilities."""

import csv
import time
from pathlib import Path
from typing import List, Dict, Any, Optional, Iterator, Callable
from collections import deque
import threading


class CSVReader:
    """Basic CSV reader."""

    def __init__(self, file_path: str):
        """
        Initialize CSV reader.

        Args:
            file_path: Path to CSV file
        """
        self.file_path = Path(file_path)
        if not self.file_path.exists():
            raise FileNotFoundError(f"CSV file not found: {file_path}")

    def read_all(self) -> List[Dict[str, Any]]:
        """
        Read all rows from the CSV file.

        Returns:
            List of dictionaries representing rows
        """
        rows = []
        with open(self.file_path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                rows.append(row)
        return rows

    def read_last_n(self, n: int) -> List[Dict[str, Any]]:
        """
        Read the last n rows from the CSV file.

        Args:
            n: Number of rows to read from the end

        Returns:
            List of dictionaries representing the last n rows
        """
        with open(self.file_path, 'r') as f:
            reader = csv.DictReader(f)
            rows = deque(reader, maxlen=n)
        return list(rows)

    def iterate_rows(self) -> Iterator[Dict[str, Any]]:
        """
        Iterate over rows in the CSV file.

        Yields:
            Dictionary representing each row
        """
        with open(self.file_path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                yield row


class CSVTailer:
    """CSV file tailer for real-time reading of growing files."""

    def __init__(
        self,
        file_path: str,
        poll_interval: float = 0.05,
        max_buffer_size: int = 20000,
    ):
        """
        Initialize CSV tailer.

        Args:
            file_path: Path to CSV file to tail
            poll_interval: Seconds between file checks
            max_buffer_size: Maximum number of rows to keep in buffer
        """
        self.file_path = Path(file_path)
        self.poll_interval = poll_interval
        self.max_buffer_size = max_buffer_size

        self.file = None
        self.reader = None
        self.header = None
        self.buffer = deque(maxlen=max_buffer_size)
        self.running = False
        self.thread = None
        self.lock = threading.Lock()

    def start(self, callback: Optional[Callable[[Dict[str, Any]], None]] = None):
        """
        Start tailing the CSV file.

        Args:
            callback: Optional function to call for each new row
        """
        if self.running:
            return

        self.running = True
        self.thread = threading.Thread(
            target=self._tail_loop,
            args=(callback,),
            daemon=True
        )
        self.thread.start()

    def stop(self):
        """Stop tailing the CSV file."""
        self.running = False
        if self.thread:
            self.thread.join(timeout=1.0)
        if self.file:
            self.file.close()
            self.file = None

    def get_buffer(self) -> List[Dict[str, Any]]:
        """
        Get the current buffer contents.

        Returns:
            List of buffered rows
        """
        with self.lock:
            return list(self.buffer)

    def _tail_loop(self, callback: Optional[Callable[[Dict[str, Any]], None]]):
        """
        Main tailing loop.

        Args:
            callback: Optional function to call for each new row
        """
        # Wait for file to exist
        while self.running and not self.file_path.exists():
            time.sleep(self.poll_interval)

        if not self.running:
            return

        # Open file and read header
        self.file = open(self.file_path, 'r')
        self.reader = csv.DictReader(self.file)
        self.header = self.reader.fieldnames

        # Read initial contents (up to buffer size)
        initial_rows = deque(self.reader, maxlen=self.max_buffer_size)
        with self.lock:
            self.buffer.extend(initial_rows)

        if callback:
            for row in initial_rows:
                callback(row)

        # Tail the file for new rows
        while self.running:
            try:
                # Check for new rows
                new_rows = []
                for row in self.reader:
                    new_rows.append(row)

                if new_rows:
                    with self.lock:
                        self.buffer.extend(new_rows)

                    if callback:
                        for row in new_rows:
                            callback(row)

                time.sleep(self.poll_interval)

            except Exception as e:
                print(f"Error tailing file: {e}")
                break

        # Clean up
        if self.file:
            self.file.close()
            self.file = None