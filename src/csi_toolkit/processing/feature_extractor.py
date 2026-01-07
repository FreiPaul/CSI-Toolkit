"""Feature extraction pipeline for CSI data."""

import csv
from pathlib import Path
from typing import List, Optional, Dict, Any

from .windowing import CSISample, WindowData, create_windows
from .features import registry, FeatureConfig


class FeatureExtractor:
    """Extract features from windowed CSI data."""

    def __init__(
        self,
        feature_names: Optional[List[str]] = None,
        labeled_mode: bool = False,
        transition_buffer: int = 1
    ):
        """
        Initialize feature extractor.

        Args:
            feature_names: List of feature names to extract (None = all features)
            labeled_mode: If True, include labels and filter transition windows
            transition_buffer: Number of windows to discard before/after transitions

        Raises:
            ValueError: If any feature name is not registered
        """
        if feature_names is None:
            self.features = registry.get_all()
        else:
            self.features = registry.get_by_names(feature_names)

        # Calculate max context windows needed
        self.max_n_prev = max(f.n_prev_windows for f in self.features) if self.features else 0
        self.max_n_next = max(f.n_next_windows for f in self.features) if self.features else 0

        # Labeled mode settings
        self.labeled_mode = labeled_mode
        self.transition_buffer = transition_buffer

    def process_file(
        self,
        input_csv: str,
        output_csv: str,
        window_size: int
    ):
        """
        Process CSV file and extract features.

        Args:
            input_csv: Path to input CSV file
            output_csv: Path to output features CSV file
            window_size: Number of samples per window

        Raises:
            FileNotFoundError: If input file doesn't exist
            ValueError: If insufficient data or invalid parameters
        """
        input_path = Path(input_csv)
        if not input_path.exists():
            raise FileNotFoundError(f"Input file not found: {input_csv}")

        print(f"Reading CSV data from {input_csv}...")
        samples = self._read_csv(input_csv)
        print(f"Loaded {len(samples)} samples")

        print(f"Creating windows (size={window_size})...")
        windows = create_windows(samples, window_size)
        print(f"Created {len(windows)} windows")

        # Check if we have enough windows
        min_windows_needed = self.max_n_prev + self.max_n_next + 1
        if len(windows) < min_windows_needed:
            raise ValueError(
                f"Insufficient windows: need at least {min_windows_needed} for "
                f"requested features, but only have {len(windows)}"
            )

        # Find windows to discard due to label transitions or label 0 (if in labeled mode)
        discard_windows = set()
        if self.labeled_mode:
            transition_windows = self._find_transition_windows(windows)
            # After-buffer must be at least max_n_prev to prevent temporal features
            # from using data from transition windows
            after_buffer = max(self.transition_buffer, self.max_n_prev)
            discard_windows = self._expand_buffer(transition_windows, len(windows), after_buffer)
            print(f"Found {len(transition_windows)} transition windows, discarding {len(discard_windows)} total (buffer: {self.transition_buffer} before, {after_buffer} after)")

            # Also discard windows with label 0 (unlabeled/transition class)
            label_zero_windows = self._find_label_zero_windows(windows)
            if label_zero_windows:
                discard_windows.update(label_zero_windows)
                print(f"Discarding {len(label_zero_windows)} additional windows with label 0")

        print(f"Extracting features (requires {self.max_n_prev} prev, {self.max_n_next} next windows)...")
        results = []
        valid_start = self.max_n_prev
        valid_end = len(windows) - self.max_n_next

        for i in range(valid_start, valid_end):
            # Skip windows with label transitions
            if i in discard_windows:
                continue

            features = self._calculate_features(windows, i)
            results.append(features)

        skipped_count = (valid_end - valid_start) - len(results)
        print(f"Calculated features for {len(results)} windows (skipped {valid_start} start, {self.max_n_next} end, {skipped_count} transitions)")

        print(f"Writing features to {output_csv}...")
        self._write_csv(output_csv, results)
        print(f"Done! Wrote {len(results)} rows")

    def _read_csv(self, file_path: str) -> List[CSISample]:
        """
        Read and parse CSV file.

        Args:
            file_path: Path to CSV file

        Returns:
            List of CSISample objects
        """
        samples = []
        with open(file_path, 'r', newline='') as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    sample = CSISample.from_csv_row(row, labeled_mode=self.labeled_mode)
                    samples.append(sample)
                except (ValueError, KeyError):
                    # Skip invalid rows silently
                    pass
        return samples

    def _find_transition_windows(self, windows: List[WindowData]) -> set:
        """
        Find windows where label transitions occur.

        A window is marked for transition if ANY sample in the window
        has a different label from the others.

        Args:
            windows: List of all windows

        Returns:
            Set of window IDs that contain label transitions
        """
        transition_windows = set()
        for window in windows:
            labels = [sample.label for sample in window.samples]
            # If more than one unique label, it's a transition window
            if len(set(labels)) > 1:
                transition_windows.add(window.window_id)
        return transition_windows

    def _find_label_zero_windows(self, windows: List[WindowData]) -> set:
        """
        Find windows where all samples have label 0.

        Label 0 is a special "unlabeled" or "transition" class used during
        data collection to mark periods between activity classes. These
        windows should be discarded as they don't represent valid activities.

        Args:
            windows: List of all windows

        Returns:
            Set of window IDs that have label 0
        """
        label_zero_windows = set()
        for window in windows:
            if window.samples and window.samples[0].label == 0:
                # Check that all samples have label 0 (should be the case
                # since transition windows are handled separately)
                labels = set(sample.label for sample in window.samples)
                if labels == {0}:
                    label_zero_windows.add(window.window_id)
        return label_zero_windows

    def _expand_buffer(self, transition_windows: set, total_windows: int, after_buffer: int) -> set:
        """
        Expand transition windows to include buffer zones.

        Uses asymmetric buffering: transition_buffer windows before,
        after_buffer windows after each transition. The after_buffer
        should be at least max_n_prev to ensure temporal features
        don't use data from discarded transition windows.

        Args:
            transition_windows: Set of window IDs with transitions
            total_windows: Total number of windows
            after_buffer: Number of windows to discard after each transition

        Returns:
            Set of all window IDs to discard (transitions + buffers)
        """
        discard = set(transition_windows)
        for win_id in transition_windows:
            # Add buffer windows before (using transition_buffer)
            for offset in range(-self.transition_buffer, 0):
                buffered_id = win_id + offset
                if 0 <= buffered_id < total_windows:
                    discard.add(buffered_id)
            # Add buffer windows after (using after_buffer)
            for offset in range(1, after_buffer + 1):
                buffered_id = win_id + offset
                if 0 <= buffered_id < total_windows:
                    discard.add(buffered_id)
        return discard

    def _calculate_features(
        self,
        windows: List[WindowData],
        window_idx: int
    ) -> Dict[str, Any]:
        """
        Calculate all features for a single window.

        Args:
            windows: All windows
            window_idx: Index of current window

        Returns:
            Dictionary with window metadata and feature values
        """
        window = windows[window_idx]

        # Start with window metadata
        row = {
            'window_id': window.window_id,
            'start_seq': window.start_seq,
            'end_seq': window.end_seq,
        }

        # Add label if in labeled mode (all samples have same label)
        if self.labeled_mode and window.samples:
            row['label'] = window.samples[0].label

        # Calculate each feature
        for feature_config in self.features:
            # Extract current window samples
            current_samples = window.samples

            # Extract previous windows' samples
            prev_samples = [
                windows[i].samples
                for i in range(
                    window_idx - feature_config.n_prev_windows,
                    window_idx
                )
            ]

            # Extract next windows' samples
            next_samples = [
                windows[i].samples
                for i in range(
                    window_idx + 1,
                    window_idx + 1 + feature_config.n_next_windows
                )
            ]

            # Calculate feature value
            try:
                value = feature_config.func(current_samples, prev_samples, next_samples)
                row[feature_config.name] = value
            except Exception as e:
                print(f"Warning: Failed to calculate {feature_config.name} for window {window_idx}: {e}")
                row[feature_config.name] = None

        return row

    def _write_csv(self, file_path: str, results: List[Dict[str, Any]]):
        """
        Write feature results to CSV.

        Args:
            file_path: Path to output CSV file
            results: List of feature dictionaries
        """
        if not results:
            print("Warning: No results to write")
            return

        # Get column names from first result
        fieldnames = list(results[0].keys())

        with open(file_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(results)

    def list_features(self) -> List[str]:
        """
        Get list of features that will be extracted.

        Returns:
            List of feature names
        """
        return [f.name for f in self.features]

    def get_feature_descriptions(self) -> Dict[str, str]:
        """
        Get descriptions of all features.

        Returns:
            Dictionary mapping feature names to descriptions
        """
        return {f.name: f.description for f in self.features}

    def extract_window_features(
        self,
        window_samples: List[CSISample],
        window_id: int = 0,
        prev_windows: Optional[List[List[CSISample]]] = None,
        next_windows: Optional[List[List[CSISample]]] = None
    ) -> Dict[str, Any]:
        """
        Extract features from a single window of samples.

        This is useful for live/real-time feature extraction where you have
        a window of samples and want to compute features immediately.

        Args:
            window_samples: List of CSISample objects for the current window
            window_id: ID to assign to this window (default: 0)
            prev_windows: List of previous windows (each is a List[CSISample])
                         Required if any features need previous context
            next_windows: List of next windows (each is a List[CSISample])
                         Required if any features need next context

        Returns:
            Dictionary with window metadata and feature values

        Raises:
            ValueError: If context windows are needed but not provided
        """
        # Create a WindowData object
        window = WindowData(
            window_id=window_id,
            start_seq=window_samples[0].seq if window_samples else 0,
            end_seq=window_samples[-1].seq if window_samples else 0,
            samples=window_samples
        )

        # Check if we need context windows
        if self.max_n_prev > 0 and (prev_windows is None or len(prev_windows) < self.max_n_prev):
            raise ValueError(
                f"Features require {self.max_n_prev} previous windows, "
                f"but only {len(prev_windows) if prev_windows else 0} provided"
            )

        if self.max_n_next > 0 and (next_windows is None or len(next_windows) < self.max_n_next):
            raise ValueError(
                f"Features require {self.max_n_next} next windows, "
                f"but only {len(next_windows) if next_windows else 0} provided"
            )

        # Build windows list for _calculate_features
        # It expects a list where window_idx points to the current window
        windows = []

        # Add previous windows (if any)
        if prev_windows:
            for prev_samples in prev_windows:
                windows.append(WindowData(
                    window_id=-1,  # Dummy ID
                    start_seq=prev_samples[0].seq if prev_samples else 0,
                    end_seq=prev_samples[-1].seq if prev_samples else 0,
                    samples=prev_samples
                ))

        # Add current window
        current_window_idx = len(windows)
        windows.append(window)

        # Add next windows (if any)
        if next_windows:
            for next_samples in next_windows:
                windows.append(WindowData(
                    window_id=-1,  # Dummy ID
                    start_seq=next_samples[0].seq if next_samples else 0,
                    end_seq=next_samples[-1].seq if next_samples else 0,
                    samples=next_samples
                ))

        # Calculate features
        return self._calculate_features(windows, current_window_idx)
