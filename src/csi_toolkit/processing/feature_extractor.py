"""Feature extraction pipeline for CSI data."""

import csv
from pathlib import Path
from typing import List, Optional, Dict, Any

from .windowing import CSISample, WindowData, create_windows
from .features import registry, FeatureConfig


class FeatureExtractor:
    """Extract features from windowed CSI data."""

    def __init__(self, feature_names: Optional[List[str]] = None):
        """
        Initialize feature extractor.

        Args:
            feature_names: List of feature names to extract (None = all features)

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

        print(f"Extracting features (requires {self.max_n_prev} prev, {self.max_n_next} next windows)...")
        results = []
        valid_start = self.max_n_prev
        valid_end = len(windows) - self.max_n_next

        for i in range(valid_start, valid_end):
            features = self._calculate_features(windows, i)
            results.append(features)

        print(f"Calculated features for {len(results)} windows (skipped {valid_start} start, {self.max_n_next} end)")

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
                    sample = CSISample.from_csv_row(row)
                    samples.append(sample)
                except (ValueError, KeyError):
                    # Skip invalid rows silently
                    pass
        return samples

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
