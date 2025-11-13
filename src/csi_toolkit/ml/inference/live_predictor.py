"""Live inference for real-time CSI data collection."""

from typing import Optional, List, Dict, Any, Callable
import numpy as np

from ..models.base import BaseModel
from ..utils import load_model_with_metadata
from ...processing.windowing import CSISample
from ...processing.feature_extractor import FeatureExtractor


class LiveInferenceHandler:
    """
    Handle live inference during CSI data collection.

    This class maintains a sliding window buffer of CSI samples and performs
    real-time predictions when complete windows are available. It uses the
    existing feature extraction and model prediction infrastructure.
    """

    def __init__(
        self,
        model_dir: str,
        window_size: int = 100,
        verbose: bool = True,
        on_prediction: Optional[Callable[[int, Any, float], None]] = None
    ):
        """
        Initialize live inference handler.

        Args:
            model_dir: Path to directory containing trained model and metadata
            window_size: Number of samples per window (should match training)
            verbose: Whether to print predictions to console
            on_prediction: Optional callback function(window_num, prediction, confidence)

        Raises:
            FileNotFoundError: If model directory or required files don't exist
            ValueError: If model or metadata are invalid
        """
        self.model_dir = model_dir
        self.window_size = window_size
        self.verbose = verbose
        self.on_prediction = on_prediction

        # Load model and metadata
        self.model: BaseModel
        self.metadata: Dict[str, Any]
        self.model, self.metadata = load_model_with_metadata(model_dir)

        # Initialize feature extractor with the same features used in training
        feature_names = self.metadata['features']
        self.feature_extractor = FeatureExtractor(
            feature_names=feature_names,
            labeled_mode=False,  # We don't need labels for live inference
            transition_buffer=0  # No transition filtering in live mode
        )

        # Initialize buffer for collecting samples
        self.buffer: List[CSISample] = []
        self.packet_count = 0
        self.window_count = 0
        self.last_prediction = None
        self.last_confidence = None

        # Maintain history of previous windows for features that need context
        self.window_history: List[List[CSISample]] = []
        self.max_history_size = self.feature_extractor.max_n_prev

        if self.verbose:
            print(f"\nLive inference initialized:")
            print(f"  Model: {self.metadata['model_type']}")
            print(f"  Classes: {self.metadata['class_names']}")
            print(f"  Features: {len(feature_names)}")
            print(f"  Window size: {window_size}")
            if self.max_history_size > 0:
                print(f"  Requires {self.max_history_size} previous windows (predictions will start after window {self.max_history_size})")
            print()

    def on_packet(self, fields: List[str]) -> Optional[Dict[str, Any]]:
        """
        Callback function for each packet collected.

        This is called by SerialCollector for every packet. It maintains a buffer
        and triggers prediction when a complete window is available.

        Args:
            fields: CSV fields from collector (including CSI data and label)

        Returns:
            Dictionary with prediction info if window complete, None otherwise:
                - 'prediction': Predicted class label
                - 'confidence': Confidence score (probability of predicted class)
                - 'window_num': Window number
                - 'probabilities': Class probabilities (optional)
        """
        try:
            # Convert fields to CSISample
            sample = self._fields_to_sample(fields)

            # Add to buffer
            self.buffer.append(sample)
            self.packet_count += 1

            # Check if we have a complete window
            if len(self.buffer) >= self.window_size:
                # Extract the window
                window_samples = self.buffer[:self.window_size]

                # Perform prediction (with previous windows if available)
                prediction_info = self._predict_window(
                    window_samples,
                    prev_windows=self.window_history[-self.max_history_size:] if self.max_history_size > 0 else None
                )

                # Add this window to history
                self.window_history.append(window_samples.copy())

                # Maintain max history size
                if len(self.window_history) > self.max_history_size:
                    self.window_history.pop(0)

                # Clear buffer (non-overlapping windows)
                self.buffer = self.buffer[self.window_size:]

                # Update tracking
                self.window_count += 1
                self.last_prediction = prediction_info['prediction']
                self.last_confidence = prediction_info['confidence']

                # Display prediction
                if self.verbose:
                    # During warmup, only display every 3rd window to reduce noise
                    if prediction_info['prediction'] != 'Unknown' or self.window_count % 3 == 0:
                        self._display_prediction(prediction_info)

                # Call callback if provided
                if self.on_prediction:
                    self.on_prediction(
                        prediction_info['window_num'],
                        prediction_info['prediction'],
                        prediction_info['confidence']
                    )

                return prediction_info

            return None

        except Exception as e:
            # If anything goes wrong, return Unknown
            if self.verbose:
                print(f"\n[Live Inference] Error: {e}")
            return {
                'prediction': 'Unknown',
                'confidence': 0.0,
                'window_num': self.window_count,
                'error': str(e)
            }

    def _fields_to_sample(self, fields: List[str]) -> CSISample:
        """
        Convert CSV row fields to CSISample object.

        Args:
            fields: List of field values from CSV row (matching CSV_HEADER format)

        Returns:
            CSISample object

        Raises:
            ValueError: If fields cannot be parsed
        """
        from ...core.constants import CSV_HEADER

        # Convert list to dictionary using CSV_HEADER
        # The fields list matches CSV_HEADER structure
        if len(fields) < len(CSV_HEADER):
            raise ValueError(f"Invalid fields length: {len(fields)}, expected at least {len(CSV_HEADER)}")

        try:
            # Create dictionary from fields
            row_dict = {header: value for header, value in zip(CSV_HEADER, fields)}

            # Use the existing CSISample.from_csv_row() method
            # This reuses all the parsing logic from the processing step
            sample = CSISample.from_csv_row(row_dict, labeled_mode=True)

            return sample

        except Exception as e:
            raise ValueError(f"Failed to convert fields to CSISample: {e}")

    def _predict_window(
        self,
        samples: List[CSISample],
        prev_windows: Optional[List[List[CSISample]]] = None
    ) -> Dict[str, Any]:
        """
        Predict label for a window of samples.

        Args:
            samples: List of CSISample objects (should be window_size length)
            prev_windows: List of previous windows (for features that need context)

        Returns:
            Dictionary with prediction info
        """
        try:
            # Check if we have enough previous windows
            if self.max_history_size > 0:
                if prev_windows is None or len(prev_windows) < self.max_history_size:
                    # Not enough history yet, return placeholder
                    return {
                        'prediction': 'Unknown',
                        'confidence': 0.0,
                        'window_num': self.window_count,
                        'error': f'Need {self.max_history_size} previous windows (have {len(prev_windows) if prev_windows else 0})'
                    }

            # Extract features using the feature extractor
            features_dict = self.feature_extractor.extract_window_features(
                window_samples=samples,
                window_id=self.window_count,
                prev_windows=prev_windows,
                next_windows=None
            )

            # Convert features to numpy array (in same order as model training)
            X = []
            for feature_name in self.metadata['features']:
                if feature_name in features_dict:
                    X.append(features_dict[feature_name])
                else:
                    # Feature missing, use 0 as default
                    X.append(0.0)

            X = np.array([X])  # Shape: (1, n_features)

            # Get prediction and probabilities
            prediction = self.model.predict(X)[0]
            probabilities = self.model.predict_proba(X)

            # Get confidence (probability of predicted class)
            if probabilities is not None and len(probabilities.shape) == 2:
                # Find which class index corresponds to the prediction
                class_names = self.metadata['class_names']
                if prediction in class_names:
                    pred_idx = class_names.index(prediction)
                    confidence = float(probabilities[0, pred_idx])
                else:
                    confidence = 0.0
            else:
                confidence = 0.0

            return {
                'prediction': prediction,
                'confidence': confidence,
                'window_num': self.window_count,
                'probabilities': probabilities[0].tolist() if probabilities is not None else None,
                'start_seq': samples[0].seq,
                'end_seq': samples[-1].seq,
            }

        except Exception as e:
            # Return Unknown on error
            return {
                'prediction': 'Unknown',
                'confidence': 0.0,
                'window_num': self.window_count,
                'error': str(e)
            }

    def _display_prediction(self, prediction_info: Dict[str, Any]):
        """
        Display prediction in console.

        Args:
            prediction_info: Dictionary with prediction details
        """
        pred = prediction_info['prediction']
        conf = prediction_info['confidence']
        win_num = prediction_info['window_num']

        if pred == 'Unknown':
            error = prediction_info.get('error', 'Unknown error')
            # Check if it's the "not enough history" case
            if 'Need' in error and 'previous windows' in error:
                print(f"[Window {win_num}] Prediction: Waiting for history... ({error})")
            else:
                print(f"[Window {win_num}] Prediction: Unknown (Error: {error})")
        else:
            print(f"[Window {win_num}] Prediction: {pred} (confidence: {conf:.2f})")

    def get_current_prediction(self) -> Optional[Any]:
        """
        Get the most recent prediction.

        Returns:
            Last predicted label, or None if no predictions yet
        """
        return self.last_prediction

    def get_current_confidence(self) -> Optional[float]:
        """
        Get confidence of most recent prediction.

        Returns:
            Last confidence score, or None if no predictions yet
        """
        return self.last_confidence

    def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics about live inference.

        Returns:
            Dictionary with inference statistics
        """
        return {
            'packets_processed': self.packet_count,
            'windows_predicted': self.window_count,
            'buffer_size': len(self.buffer),
            'last_prediction': self.last_prediction,
            'last_confidence': self.last_confidence,
        }
