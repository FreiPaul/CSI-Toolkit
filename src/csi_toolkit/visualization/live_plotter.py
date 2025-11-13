"""Live plotting functionality for CSI data."""

import time
import threading
from typing import Optional, List
from collections import deque
import matplotlib

# Try to set a backend that works on the current system
# Priority: MacOSX (native on macOS) > TkAgg > Qt5Agg > automatic
try:
    import platform
    if platform.system() == 'Darwin':  # macOS
        matplotlib.use('MacOSX')
    else:
        # Try TkAgg for other platforms
        try:
            matplotlib.use('TkAgg')
        except:
            pass  # Let matplotlib choose automatically
except:
    pass  # Let matplotlib use default backend

import matplotlib.pyplot as plt
import matplotlib.animation as animation

from ..core.parser import parse_amplitude_json
from ..processing.amplitude import compute_mean_amplitude
from ..io.csv_reader import CSVTailer
from ..io.ssh_reader import SSHReader
from .filters import apply_filter


class LivePlotter:
    """Real-time plotter for CSI amplitude data."""

    def __init__(
        self,
        file_path: str,
        subcarrier: int = 10,
        refresh_rate: float = 0.2,
        max_points: int = 20000,
        display_limit: Optional[int] = None,
        filter_type: str = 'moving_average',
        filter_params: Optional[dict] = None,
    ):
        """
        Initialize live plotter.

        Args:
            file_path: Path to CSV file (local or SSH format)
            subcarrier: Subcarrier index to plot
            refresh_rate: Plot update interval in seconds
            max_points: Maximum points to keep in memory
            display_limit: Number of points to display (None for all)
            filter_type: Type of filter to apply
            filter_params: Parameters for the filter
        """
        self.file_path = file_path
        self.subcarrier = subcarrier
        self.refresh_rate = refresh_rate
        self.max_points = max_points
        self.display_limit = display_limit or 500  # Default display limit
        self.filter_type = filter_type
        self.filter_params = filter_params or {}

        # Data buffers
        self.mean_amplitudes = deque(maxlen=max_points)
        self.subcarrier_amplitudes = deque(maxlen=max_points)
        self.sequence_numbers = deque(maxlen=max_points)

        # Threading
        self.lock = threading.Lock()
        self.running = False
        self.reader = None

        # Matplotlib setup
        self.fig = None
        self.ax1 = None
        self.ax2 = None
        self.line1_raw = None
        self.line1_filtered = None
        self.line2_raw = None
        self.line2_filtered = None
        self.ani = None

    def start(self):
        """Start live plotting."""
        if self.running:
            return

        self.running = True

        # Determine if SSH or local file
        if '@' in self.file_path and ':' in self.file_path:
            self.reader = SSHReader(
                self.file_path,
                poll_interval=0.05,
                max_buffer_size=self.max_points
            )
        else:
            self.reader = CSVTailer(
                self.file_path,
                poll_interval=0.05,
                max_buffer_size=self.max_points
            )

        # Start reader with callback
        self.reader.start(callback=self._process_row)

        # Setup plot
        self._setup_plot()

        # Use matplotlib animation for thread-safe updates
        self.ani = animation.FuncAnimation(
            self.fig,
            self._update_plot_animation,
            interval=int(self.refresh_rate * 1000),  # Convert to milliseconds
            blit=False,
            cache_frame_data=False
        )

        try:
            plt.show()
        except KeyboardInterrupt:
            print("\nPlotting stopped by user")
        finally:
            self.stop()

    def stop(self):
        """Stop live plotting."""
        self.running = False

        if self.reader:
            self.reader.stop()

        plt.close('all')

    def _process_row(self, row: dict):
        """
        Process a new row from the CSV file.

        Args:
            row: Dictionary containing CSV row data
        """
        try:
            # Extract sequence number
            seq = int(row.get('seq', 0))

            # Extract and parse amplitudes
            amplitudes_str = row.get('amplitudes', '')
            if not amplitudes_str or amplitudes_str == '[]':
                return

            # Parse amplitude JSON
            amplitude_values = parse_amplitude_json(amplitudes_str)
            if not amplitude_values:
                return

            # Compute mean amplitude
            mean_amp = compute_mean_amplitude(amplitude_values)

            # Get specific subcarrier amplitude
            if 0 <= self.subcarrier < len(amplitude_values):
                subcarrier_amp = amplitude_values[self.subcarrier]
            else:
                subcarrier_amp = 0.0

            # Store in buffers (thread-safe)
            with self.lock:
                self.sequence_numbers.append(seq)
                self.mean_amplitudes.append(mean_amp)
                self.subcarrier_amplitudes.append(subcarrier_amp)

        except Exception:
            # Silently skip bad rows
            pass

    def _setup_plot(self):
        """Setup matplotlib figure and axes."""
        self.fig, (self.ax1, self.ax2) = plt.subplots(2, 1, figsize=(12, 8))

        # Setup top plot (mean amplitude)
        self.ax1.set_title('Mean Amplitude Across All Subcarriers')
        self.ax1.set_xlabel('Sequence Number')
        self.ax1.set_ylabel('Amplitude')
        self.ax1.grid(True, alpha=0.3)
        self.line1_raw, = self.ax1.plot([], [], 'b-', alpha=0.3, label='Raw')
        self.line1_filtered, = self.ax1.plot([], [], 'r-', label='Filtered')
        self.ax1.legend(loc='upper right')

        # Setup bottom plot (single subcarrier)
        self.ax2.set_title(f'Subcarrier {self.subcarrier} Amplitude')
        self.ax2.set_xlabel('Sequence Number')
        self.ax2.set_ylabel('Amplitude')
        self.ax2.grid(True, alpha=0.3)
        self.line2_raw, = self.ax2.plot([], [], 'b-', alpha=0.3, label='Raw')
        self.line2_filtered, = self.ax2.plot([], [], 'r-', label='Filtered')
        self.ax2.legend(loc='upper right')

        plt.tight_layout()

    def _update_plot_animation(self, frame):
        """
        Update plot (called by FuncAnimation).

        Args:
            frame: Frame number (not used)
        """
        # Get data from buffers (thread-safe)
        with self.lock:
            seq_nums = list(self.sequence_numbers)
            mean_amps = list(self.mean_amplitudes)
            sub_amps = list(self.subcarrier_amplitudes)

        if not seq_nums:
            return self.line1_raw, self.line1_filtered, self.line2_raw, self.line2_filtered

        # Apply display limit
        if len(seq_nums) > self.display_limit:
            seq_nums = seq_nums[-self.display_limit:]
            mean_amps = mean_amps[-self.display_limit:]
            sub_amps = sub_amps[-self.display_limit:]

        # Apply filtering
        if len(mean_amps) > 1:
            filtered_mean = apply_filter(
                mean_amps,
                self.filter_type,
                **self.filter_params
            )
            filtered_sub = apply_filter(
                sub_amps,
                self.filter_type,
                **self.filter_params
            )
        else:
            filtered_mean = mean_amps
            filtered_sub = sub_amps

        # Update plot data
        self.line1_raw.set_data(seq_nums, mean_amps)
        self.line1_filtered.set_data(seq_nums, filtered_mean)
        self.line2_raw.set_data(seq_nums, sub_amps)
        self.line2_filtered.set_data(seq_nums, filtered_sub)

        # Adjust axes limits
        for ax, raw_data, filtered_data in [
            (self.ax1, mean_amps, filtered_mean),
            (self.ax2, sub_amps, filtered_sub)
        ]:
            ax.relim()
            ax.autoscale_view()

            # Set y-limits with some padding
            all_data = raw_data + filtered_data
            if all_data:
                y_min, y_max = min(all_data), max(all_data)
                y_range = y_max - y_min
                if y_range > 0:
                    ax.set_ylim(y_min - 0.1 * y_range, y_max + 0.1 * y_range)

        return self.line1_raw, self.line1_filtered, self.line2_raw, self.line2_filtered