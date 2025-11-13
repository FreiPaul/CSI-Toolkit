"""SSH-based CSV reader for remote files."""

import subprocess
import csv
import time
import re
from typing import List, Dict, Any, Optional, Callable, Tuple
from collections import deque
import threading
import io


class SSHReader:
    """Read and tail CSV files from remote hosts via SSH."""

    def __init__(
        self,
        ssh_path: str,
        poll_interval: float = 0.05,
        max_buffer_size: int = 20000,
        timeout: int = 10,
    ):
        """
        Initialize SSH reader.

        Args:
            ssh_path: SSH path in format user@host:/path/to/file
            poll_interval: Seconds between file checks
            max_buffer_size: Maximum number of rows to keep in buffer
            timeout: SSH command timeout in seconds
        """
        self.poll_interval = poll_interval
        self.max_buffer_size = max_buffer_size
        self.timeout = timeout

        # Parse SSH path
        self.user, self.host, self.remote_path = self._parse_ssh_path(ssh_path)

        self.buffer = deque(maxlen=max_buffer_size)
        self.running = False
        self.thread = None
        self.lock = threading.Lock()
        self.byte_position = 0
        self.header = None
        self.partial_line = ""

    def _parse_ssh_path(self, ssh_path: str) -> Tuple[str, str, str]:
        """
        Parse SSH path format.

        Args:
            ssh_path: Path in format user@host:/path/to/file

        Returns:
            Tuple of (user, host, remote_path)
        """
        match = re.match(r'([^@]+)@([^:]+):(.+)', ssh_path)
        if not match:
            raise ValueError(f"Invalid SSH path format: {ssh_path}")
        return match.groups()

    def _run_ssh_command(self, command: str) -> str:
        """
        Run an SSH command and return output.

        Args:
            command: Command to run on remote host

        Returns:
            Command output as string
        """
        ssh_cmd = [
            'ssh',
            f'{self.user}@{self.host}',
            command
        ]

        try:
            result = subprocess.run(
                ssh_cmd,
                capture_output=True,
                text=True,
                timeout=self.timeout,
                check=True
            )
            return result.stdout
        except subprocess.TimeoutExpired:
            raise TimeoutError(f"SSH command timed out: {command}")
        except subprocess.CalledProcessError as e:
            raise ConnectionError(f"SSH command failed: {e.stderr}")

    def read_all(self) -> List[Dict[str, Any]]:
        """
        Read all rows from the remote CSV file.

        Returns:
            List of dictionaries representing rows
        """
        # Use cat to read entire file
        content = self._run_ssh_command(f'cat "{self.remote_path}"')

        # Parse CSV content
        rows = []
        reader = csv.DictReader(io.StringIO(content))
        for row in reader:
            rows.append(row)

        return rows

    def read_last_n(self, n: int) -> List[Dict[str, Any]]:
        """
        Read the last n rows from the remote CSV file.

        Args:
            n: Number of rows to read from the end

        Returns:
            List of dictionaries representing the last n rows
        """
        # Use tail to get last n+1 lines (including header)
        # Then prepend header if needed
        header_cmd = f'head -1 "{self.remote_path}"'
        tail_cmd = f'tail -n {n} "{self.remote_path}"'

        header = self._run_ssh_command(header_cmd).strip()
        tail_content = self._run_ssh_command(tail_cmd)

        # Combine header and tail content
        full_content = header + '\n' + tail_content

        # Parse CSV content
        rows = []
        reader = csv.DictReader(io.StringIO(full_content))
        for row in reader:
            rows.append(row)

        return rows

    def start(self, callback: Optional[Callable[[Dict[str, Any]], None]] = None):
        """
        Start tailing the remote CSV file.

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
        """Stop tailing the remote CSV file."""
        self.running = False
        if self.thread:
            self.thread.join(timeout=1.0)

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
        Main tailing loop for remote file.

        Args:
            callback: Optional function to call for each new row
        """
        # Get initial file size and header
        try:
            # Get file size
            size_output = self._run_ssh_command(f'stat -c %s "{self.remote_path}" 2>/dev/null || stat -f %z "{self.remote_path}"')
            file_size = int(size_output.strip())

            # Read header
            header_output = self._run_ssh_command(f'head -1 "{self.remote_path}"')
            self.header = header_output.strip().split(',')

            # Read initial content (last part of file)
            if file_size > 1000000:  # If file is > 1MB, only read last 1MB
                self.byte_position = file_size - 1000000
                initial_cmd = f'tail -c 1000000 "{self.remote_path}"'
            else:
                self.byte_position = 0
                initial_cmd = f'cat "{self.remote_path}"'

            initial_content = self._run_ssh_command(initial_cmd)

            # Parse initial content
            reader = csv.DictReader(io.StringIO(initial_content))
            initial_rows = list(reader)[-self.max_buffer_size:]  # Keep last n rows

            with self.lock:
                self.buffer.extend(initial_rows)

            if callback:
                for row in initial_rows:
                    callback(row)

            # Update byte position
            self.byte_position = file_size

        except Exception as e:
            print(f"Error initializing SSH tail: {e}")
            return

        # Tail the file for new content
        while self.running:
            try:
                # Get current file size
                size_output = self._run_ssh_command(f'stat -c %s "{self.remote_path}" 2>/dev/null || stat -f %z "{self.remote_path}"')
                current_size = int(size_output.strip())

                # Check if file has grown
                if current_size > self.byte_position:
                    # Read new bytes
                    bytes_to_read = current_size - self.byte_position
                    new_content_cmd = f'tail -c +{self.byte_position + 1} "{self.remote_path}" | head -c {bytes_to_read}'
                    new_content = self._run_ssh_command(new_content_cmd)

                    # Handle partial lines
                    full_content = self.partial_line + new_content
                    lines = full_content.split('\n')

                    # Save last partial line for next iteration
                    if not full_content.endswith('\n'):
                        self.partial_line = lines[-1]
                        lines = lines[:-1]
                    else:
                        self.partial_line = ""

                    # Parse new rows
                    if lines and lines[0]:  # Skip empty lines
                        # Create CSV content with header
                        csv_content = ','.join(self.header) + '\n' + '\n'.join(lines)
                        reader = csv.DictReader(io.StringIO(csv_content))
                        new_rows = list(reader)

                        if new_rows:
                            with self.lock:
                                self.buffer.extend(new_rows)

                            if callback:
                                for row in new_rows:
                                    callback(row)

                    # Update byte position
                    self.byte_position = current_size

                time.sleep(self.poll_interval)

            except Exception as e:
                print(f"Error in SSH tail loop: {e}")
                # Wait before retrying
                time.sleep(1.0)