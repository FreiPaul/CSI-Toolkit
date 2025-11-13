"""Serial data collector for CSI data."""

import serial
import signal
import sys
import time
from datetime import datetime
from typing import Optional, List, Callable

from .config import CollectorConfig
from ..core.parser import parse_csi_line, parse_amplitude_json
from ..core.constants import CSV_HEADER
from ..processing.amplitude import calculate_amplitudes
from ..io.csv_writer import CSVWriter


class SerialCollector:
    """Collect CSI data from serial port and save to CSV."""

    def __init__(
        self,
        config: CollectorConfig,
        callback: Optional[Callable[[List], None]] = None,
        debug: bool = False,
    ):
        """
        Initialize serial collector.

        Args:
            config: Collector configuration
            callback: Optional callback for each processed row
            debug: Enable debug output for troubleshooting
        """
        self.config = config
        self.callback = callback
        self.debug = debug

        self.serial_port = None
        self.csv_writer = None
        self.running = False
        self.packet_count = 0
        self.error_count = 0
        self.startup_packets = 0  # Track initial packets for debugging

        # Setup signal handlers
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

    def _signal_handler(self, signum, frame):
        """Handle shutdown signals gracefully."""
        print("\nShutdown signal received. Cleaning up...")
        self.stop()

    def start(self):
        """Start collecting data from serial port."""
        if self.running:
            return

        self.running = True
        print(f"Starting CSI data collection")
        print(self.config)

        try:
            # Open serial port
            self._open_serial()

            # Open CSV writer
            self.csv_writer = CSVWriter(
                output_dir=self.config.output_dir,
                flush_interval=self.config.flush_interval,
                header=CSV_HEADER,
            )
            self.csv_writer.open()

            # Start collection loop
            self._collection_loop()

        except Exception as e:
            print(f"Collection error: {e}")
            raise

        finally:
            self.stop()

    def stop(self):
        """Stop data collection and clean up resources."""
        self.running = False

        # Close serial port
        if self.serial_port and self.serial_port.is_open:
            self.serial_port.close()
            print("Serial port closed")

        # Close CSV writer
        if self.csv_writer:
            saved_path = self.csv_writer.close()
            print(f"CSV file saved: {saved_path}")
            print(f"Total packets collected: {self.packet_count}")
            print(f"Total errors: {self.error_count}")

    def _open_serial(self):
        """Open serial port connection."""
        try:
            self.serial_port = serial.Serial(
                port=self.config.serial_port,
                baudrate=self.config.baudrate,
                timeout=1.0
            )

            # Clear any buffered data
            self.serial_port.reset_input_buffer()
            print(f"Connected to {self.config.serial_port} at {self.config.baudrate} baud")

        except serial.SerialException as e:
            raise ConnectionError(f"Failed to open serial port: {e}")

    def _collection_loop(self):
        """Main collection loop."""
        print("Collecting data... Press Ctrl+C to stop")

        while self.running:
            try:
                # Read line from serial
                if not self.serial_port or not self.serial_port.is_open:
                    break

                line_bytes = self.serial_port.readline()
                if not line_bytes:
                    continue

                # Decode line
                try:
                    line = line_bytes.decode('utf-8', errors='ignore').strip()
                except UnicodeDecodeError:
                    self.error_count += 1
                    continue

                # Process CSI data line
                self._process_line(line)

            except serial.SerialException as e:
                print(f"Serial error: {e}")
                self.error_count += 1
                # Try to reconnect
                time.sleep(1.0)
                try:
                    self._open_serial()
                except:
                    break

            except KeyboardInterrupt:
                break

            except Exception as e:
                print(f"Unexpected error: {e}")
                self.error_count += 1

    def _process_line(self, line: str):
        """
        Process a line from serial input.

        Args:
            line: Raw line from serial port
        """
        # Parse CSI line
        fields = parse_csi_line(line)
        if not fields:
            return  # Not a CSI_DATA line

        # Get current timestamp
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
        fields[9] = timestamp  # Update local_timestamp field

        # Extract and process amplitudes
        try:
            # Get the data field (JSON array)
            data_field = fields[13] if len(fields) > 13 else ""

            if data_field and data_field != "[]":
                # Parse Q,I values
                q_i_values = parse_amplitude_json(data_field)

                # Only try to calculate amplitudes if we have valid data
                if q_i_values and len(q_i_values) > 1:
                    # Calculate amplitudes
                    amplitudes = calculate_amplitudes(q_i_values)
                    # Convert to JSON string for CSV
                    amplitudes_str = str(amplitudes)
                else:
                    # Data exists but is too short or malformed
                    amplitudes_str = "[]"
                    if q_i_values:  # Only log if we got some data but it's wrong
                        if self.debug or self.startup_packets < 5:
                            print(f"Warning: Incomplete Q,I data (got {len(q_i_values)} values)")
                            if self.debug:
                                print(f"  Raw data field: {data_field[:100]}...")
                        self.error_count += 1
            else:
                amplitudes_str = "[]"

            # Append amplitudes to fields
            fields.append(amplitudes_str)

        except Exception as e:
            # Only print detailed errors occasionally to avoid spam
            if self.error_count < 10 or self.error_count % 100 == 0:
                print(f"Amplitude processing error #{self.error_count + 1}: {e}")
                if self.error_count == 10:
                    print("(Further errors will be shown every 100 occurrences)")
            fields.append("[]")  # Empty amplitudes on error
            self.error_count += 1

        # Prepend type field for CSV
        full_row = ['CSI_DATA'] + fields

        # Write to CSV
        self.csv_writer.write_row(full_row)
        self.packet_count += 1
        self.startup_packets += 1

        # Optional callback
        if self.callback:
            self.callback(fields)

        # Print progress
        if self.packet_count % 100 == 0:
            print(f"Packets collected: {self.packet_count}, Errors: {self.error_count}", end='\r')

    def get_statistics(self) -> dict:
        """
        Get collection statistics.

        Returns:
            Dictionary of statistics
        """
        return {
            'packet_count': self.packet_count,
            'error_count': self.error_count,
            'error_rate': self.error_count / max(1, self.packet_count),
        }