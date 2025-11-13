"""Serial data collector for CSI data."""

import serial
import signal
import sys
import time
from datetime import datetime
from typing import Optional, List, Callable

from pynput import keyboard

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
        live_inference_handler=None,  # LiveInferenceHandler instance
    ):
        """
        Initialize serial collector.

        Args:
            config: Collector configuration
            callback: Optional callback for each processed row
            debug: Enable debug output for troubleshooting
            live_inference_handler: Optional LiveInferenceHandler for real-time predictions
        """
        self.config = config
        self.callback = callback
        self.debug = debug
        self.live_inference_handler = live_inference_handler

        self.serial_port = None
        self.csv_writer = None
        self.running = False
        self.packet_count = 0
        self.error_count = 0
        self.startup_packets = 0  # Track initial packets for debugging

        # Labeling support
        self.current_label = 0  # Default: 0 = unlabeled
        self.keyboard_listener = None

        # Live inference tracking
        self.current_prediction = None
        self.current_confidence = None

        # Setup signal handlers
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

    def _signal_handler(self, signum, frame):
        """Handle shutdown signals gracefully."""
        print("\nShutdown signal received. Cleaning up...")
        self.stop()

    def _on_key_press(self, key):
        """
        Handle keyboard input for labeling.

        Keys 0-9 set the current label (0 = unlabeled, 1-9 = class labels).

        Args:
            key: Key pressed (from pynput)
        """
        try:
            # Check if it's a character key
            if hasattr(key, 'char') and key.char is not None and key.char in '0123456789':
                new_label = int(key.char)
                if new_label != self.current_label:
                    self.current_label = new_label
                    label_name = "unlabeled" if new_label == 0 else f"class {new_label}"
                    print(f"\n[LABEL] Changed to: {label_name} ({new_label})")
        except Exception as e:
            # Silently ignore errors (e.g., special keys)
            pass

    def start(self):
        """Start collecting data from serial port."""
        if self.running:
            return

        self.running = True
        print(f"Starting CSI data collection")
        print(self.config)
        print("\n[LABELING] Press keys 0-9 to set label (0=unlabeled, 1-9=classes)")
        print(f"[LABELING] Current label: {self.current_label} (unlabeled)")

        try:
            # Start keyboard listener
            self.keyboard_listener = keyboard.Listener(on_press=self._on_key_press)
            self.keyboard_listener.start()

            # Open serial port
            self._open_serial()

            # Prepare CSV header (add predicted_label if live inference is enabled)
            header = CSV_HEADER.copy()
            if self.live_inference_handler:
                header.append('predicted_label')
                print(f"[LIVE INFERENCE] Enabled - predictions will be saved to CSV")

            # Open CSV writer
            self.csv_writer = CSVWriter(
                output_dir=self.config.output_dir,
                flush_interval=self.config.flush_interval,
                header=header,
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

        # Stop keyboard listener
        if self.keyboard_listener:
            self.keyboard_listener.stop()
            self.keyboard_listener = None

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

        # Prepend type field and append label for CSV
        full_row = ['CSI_DATA'] + fields + [str(self.current_label)]

        # Live inference (if enabled)
        if self.live_inference_handler:
            # Call the live inference handler
            prediction_info = self.live_inference_handler.on_packet(full_row)

            # Update current prediction if window was complete
            if prediction_info:
                self.current_prediction = prediction_info.get('prediction', 'Unknown')
                self.current_confidence = prediction_info.get('confidence', 0.0)

            # Add predicted label to the row (use current prediction or 'Unknown')
            pred_label = self.current_prediction if self.current_prediction is not None else 'Unknown'
            full_row.append(str(pred_label))

        # Write to CSV
        self.csv_writer.write_row(full_row)
        self.packet_count += 1
        self.startup_packets += 1

        # Optional callback (separate from live inference)
        if self.callback:
            self.callback(fields)

        # Print progress with prediction if available
        if self.packet_count % 100 == 0:
            if self.live_inference_handler and self.current_prediction is not None:
                conf_str = f" ({self.current_confidence:.2f})" if self.current_confidence else ""
                print(f"Packets: {self.packet_count}, Errors: {self.error_count} | Prediction: {self.current_prediction}{conf_str}", end='\r')
            else:
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