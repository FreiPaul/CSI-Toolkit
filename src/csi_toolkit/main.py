#!/usr/bin/env python3
"""Main entry point for CSI Toolkit with CLI interface."""

import argparse
import sys
from pathlib import Path


def collect_command(args):
    """Handle the collect subcommand."""
    from .collection import SerialCollector, CollectorConfig

    # Create configuration
    config = CollectorConfig(
        serial_port=args.port,
        baudrate=args.baudrate,
        flush_interval=args.flush,
        output_dir=args.output_dir,
        env_file=args.env,
    )

    # Setup live inference if requested
    live_inference_handler = None
    if args.live_inference:
        if not args.model_dir:
            print("Error: --live-inference requires --model-dir to be specified", file=sys.stderr)
            return 1

        try:
            from .ml.inference import LiveInferenceHandler
            live_inference_handler = LiveInferenceHandler(
                model_dir=args.model_dir,
                window_size=args.window_size,
                verbose=True,
            )
        except ImportError:
            print("Error: ML functionality not installed. Install with: pip install -e '.[ml]'", file=sys.stderr)
            return 1
        except Exception as e:
            print(f"Error loading model: {e}", file=sys.stderr)
            return 1

    # Create and start collector
    collector = SerialCollector(
        config,
        debug=args.debug,
        live_inference_handler=live_inference_handler,
    )
    try:
        collector.start()
    except KeyboardInterrupt:
        print("\nCollection stopped by user")
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    return 0


def plot_command(args):
    """Handle the plot subcommand."""
    from .visualization import LivePlotter

    # Prepare filter parameters
    filter_params = {
        'window_size': args.window_size,
    }

    # Add Butterworth filter params if sampling rate is specified
    if args.fs:
        filter_params['sampling_rate'] = args.fs
        filter_params['cutoff_freq'] = args.fc
        filter_params['order'] = args.order
        filter_type = 'butterworth'
    else:
        filter_type = 'moving_average'

    # Create and start plotter
    plotter = LivePlotter(
        file_path=args.file,
        subcarrier=args.subcarrier,
        refresh_rate=args.refresh,
        max_points=args.maxpoints,
        display_limit=args.limit,
        filter_type=filter_type,
        filter_params=filter_params,
    )

    try:
        # Start will now block until the window is closed
        plotter.start()
    except KeyboardInterrupt:
        print("\nPlotting stopped by user")
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1

    return 0


def process_command(args):
    """Handle the process subcommand."""
    from .processing import FeatureExtractor
    from .processing.feature_extractor import stratified_split
    from .processing.features import registry

    # Handle --list-features first (doesn't need input/output files)
    if args.list_features:
        print("Available features:")
        print()
        for config in registry.get_all():
            n_context = []
            if config.n_prev_windows > 0:
                n_context.append(f"{config.n_prev_windows} prev")
            if config.n_next_windows > 0:
                n_context.append(f"{config.n_next_windows} next")
            context_str = f" ({', '.join(n_context)})" if n_context else ""
            print(f"  {config.name:15s} - {config.description}{context_str}")
        return 0

    # Validate input and output are provided
    if not args.input or not args.output:
        print("Error: input and output files are required", file=sys.stderr)
        print("Use --list-features to see available features", file=sys.stderr)
        return 1

    # Validate split option
    if args.split is not None:
        if not args.labeled:
            print("Error: --split requires --labeled mode (need labels for stratified split)", file=sys.stderr)
            return 1
        if not 1 <= args.split <= 99:
            print("Error: --split must be between 1 and 99", file=sys.stderr)
            return 1

    # Parse feature names if provided
    feature_names = None
    if args.features:
        feature_names = [f.strip() for f in args.features.split(',')]

    # Create feature extractor
    try:
        extractor = FeatureExtractor(
            feature_names,
            labeled_mode=args.labeled,
            transition_buffer=args.transition_buffer
        )
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        print(f"\nAvailable features: {', '.join(registry.list_names())}")
        return 1

    # Process file
    try:
        if args.split is not None:
            # Extract features without writing (we'll split and write separately)
            results = extractor.extract_features(
                input_csv=args.input,
                window_size=args.window_size,
            )

            # Perform stratified split by label
            train_results, test_results = stratified_split(results, args.split / 100.0)

            # Generate output filenames with -train and -test suffixes
            output_path = Path(args.output)
            stem = output_path.stem
            suffix = output_path.suffix or '.csv'
            parent = output_path.parent

            train_path = parent / f"{stem}-train{suffix}"
            test_path = parent / f"{stem}-test{suffix}"

            # Write both files
            print(f"Writing train features to {train_path}...")
            extractor.write_results(str(train_path), train_results)
            print(f"Done! Wrote {len(train_results)} rows to train set")

            print(f"Writing test features to {test_path}...")
            extractor.write_results(str(test_path), test_results)
            print(f"Done! Wrote {len(test_results)} rows to test set")

            # Print split summary per label
            _print_split_summary(results, train_results, test_results)
        else:
            extractor.process_file(
                input_csv=args.input,
                output_csv=args.output,
                window_size=args.window_size,
            )
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1

    return 0


def _print_split_summary(all_results, train_results, test_results):
    """Print a summary of the stratified split per label."""
    from collections import Counter

    all_labels = Counter(r.get('label') for r in all_results)
    train_labels = Counter(r.get('label') for r in train_results)
    test_labels = Counter(r.get('label') for r in test_results)

    print("\nSplit summary per label:")
    print(f"  {'Label':<10} {'Total':>8} {'Train':>8} {'Test':>8} {'Train%':>8}")
    print(f"  {'-'*10} {'-'*8} {'-'*8} {'-'*8} {'-'*8}")
    for label in sorted(all_labels.keys()):
        total = all_labels[label]
        train = train_labels.get(label, 0)
        test = test_labels.get(label, 0)
        pct = (train / total * 100) if total > 0 else 0
        print(f"  {label:<10} {total:>8} {train:>8} {test:>8} {pct:>7.1f}%")
    print(f"  {'-'*10} {'-'*8} {'-'*8} {'-'*8} {'-'*8}")
    total_all = len(all_results)
    total_train = len(train_results)
    total_test = len(test_results)
    total_pct = (total_train / total_all * 100) if total_all > 0 else 0
    print(f"  {'Total':<10} {total_all:>8} {total_train:>8} {total_test:>8} {total_pct:>7.1f}%")


def train_command(args):
    """Handle the train subcommand."""
    from .ml import ModelTrainer, model_registry

    # Handle --list-models first
    if args.list_models:
        print(model_registry.list_models())
        return 0

    # Validate input is provided
    if not args.input:
        print("Error: input file is required", file=sys.stderr)
        print("Use --list-models to see available model types", file=sys.stderr)
        return 1

    # Parse model parameters if provided
    model_params = {}
    if args.params:
        try:
            # Parse key=value pairs
            for param_str in args.params.split(','):
                key, value = param_str.split('=')
                # Try to convert to appropriate type
                try:
                    value = int(value)
                except ValueError:
                    try:
                        value = float(value)
                    except ValueError:
                        pass  # Keep as string
                model_params[key.strip()] = value
        except ValueError:
            print("Error: Invalid parameter format. Use key=value,key=value", file=sys.stderr)
            return 1

    # Create trainer
    try:
        trainer = ModelTrainer(
            model_type=args.model,
            train_split=args.train_split,
            val_split=args.val_split,
            test_split=args.test_split,
            random_seed=args.seed,
            model_params=model_params
        )
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    # Train model
    try:
        output_dir = trainer.train(
            input_csv=args.input,
            output_dir=args.output_dir
        )
        print(f"\n✓ Training completed successfully!")
        print(f"Model saved to: {output_dir}")
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1

    return 0


def inference_command(args):
    """Handle the inference subcommand."""
    from .ml import ModelPredictor

    # Validate required arguments
    if not args.dataset or not args.model_dir:
        print("Error: --dataset and --model-dir are required", file=sys.stderr)
        return 1

    # Create predictor
    try:
        predictor = ModelPredictor(model_dir=args.model_dir)
    except Exception as e:
        print(f"Error loading model: {e}", file=sys.stderr)
        return 1

    # Run inference
    try:
        output_csv = predictor.run_inference(
            input_csv=args.dataset,
            output_csv=args.output,
            include_probabilities=args.probabilities
        )
        print(f"\n✓ Inference completed successfully!")
        print(f"Predictions saved to: {output_csv}")
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1

    return 0


def evaluate_command(args):
    """Handle the evaluate subcommand."""
    from .ml import ModelEvaluator, metric_registry

    # Handle --list-metrics first
    if args.list_metrics:
        print(metric_registry.list_metrics())
        return 0

    # Validate required arguments
    if not args.dataset or not args.model_dir:
        print("Error: --dataset and --model-dir are required", file=sys.stderr)
        print("Use --list-metrics to see available metrics", file=sys.stderr)
        return 1

    # Parse metric names if provided
    metric_names = None
    if args.metrics:
        metric_names = [m.strip() for m in args.metrics.split(',')]

    # Create evaluator
    try:
        evaluator = ModelEvaluator(
            model_dir=args.model_dir,
            metric_names=metric_names
        )
    except Exception as e:
        print(f"Error loading model: {e}", file=sys.stderr)
        return 1

    # Run evaluation
    try:
        metrics = evaluator.run_evaluation(
            input_csv=args.dataset,
            output_json=args.output_json,
            output_txt=args.output_txt
        )
        print(f"\n✓ Evaluation completed successfully!")
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1

    return 0


def plot_data_command(args):
    """Handle the plot-data subcommand."""
    from .visualization import DataPlotter, plot_registry

    # Handle --list-plots first
    if args.list_plots:
        print(plot_registry.list_plots())
        return 0

    # Validate input is provided
    if not args.input:
        print("Error: input file is required", file=sys.stderr)
        print("Use --list-plots to see available plot types", file=sys.stderr)
        return 1

    # Parse plot names if provided
    plot_names = None
    if args.plots:
        plot_names = [p.strip() for p in args.plots.split(',')]

    # Create plotter and generate plots
    try:
        plotter = DataPlotter(args.input, display=args.display)
        output_paths = plotter.generate_plots(plot_names=plot_names)

        if output_paths:
            print(f"\n✓ Generated {len(output_paths)} plot(s)")
        else:
            return 1

    except FileNotFoundError:
        print(f"Error: File not found: {args.input}", file=sys.stderr)
        return 1
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1

    return 0


def info_command(args):
    """Handle the info subcommand."""
    print("CSI Toolkit - Channel State Information Processing Pipeline")
    print("=" * 60)
    print("\nModules:")
    print("  - Collection: Serial data acquisition from ESP32")
    print("  - Visualization: Real-time plotting with filtering")
    print("  - Processing: Signal processing and feature extraction")
    print("  - Machine Learning: Model training, inference, and evaluation")
    print("  - I/O: CSV reading/writing with SSH support")
    print("\nData Flow:")
    print("  ESP32 -> Serial -> CSV -> Processing -> ML Training -> Inference")
    print("\nML Workflow:")
    print("  1. Collect labeled data: csi_toolkit collect")
    print("  2. Extract features: csi_toolkit process --labeled")
    print("  3. Train model: csi_toolkit train features.csv")
    print("  4. Evaluate model: csi_toolkit evaluate --dataset test.csv --model-dir models/model_X")
    print("  5. Run inference: csi_toolkit inference --dataset new.csv --model-dir models/model_X")
    print("\nFor detailed help on each command, use:")
    print("  python -m csi_toolkit <command> --help")
    return 0


def main():
    """Main entry point for CLI."""
    parser = argparse.ArgumentParser(
        description='CSI Toolkit - Modular pipeline for CSI data processing',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    subparsers = parser.add_subparsers(
        dest='command',
        help='Available commands',
        required=True,
    )

    # Collect command
    collect_parser = subparsers.add_parser(
        'collect',
        help='Collect CSI data from serial port',
        description='Collect CSI data from ESP32 device via serial port',
    )
    collect_parser.add_argument(
        '-p', '--port',
        help='Serial port (default: from .env or /dev/cu.usbmodem1101)',
        default=None,
    )
    collect_parser.add_argument(
        '-b', '--baudrate',
        type=int,
        help='Baud rate (default: from .env or 921600)',
        default=None,
    )
    collect_parser.add_argument(
        '-f', '--flush',
        type=int,
        help='Flush interval in packets (default: from .env or 1)',
        default=None,
    )
    collect_parser.add_argument(
        '-o', '--output-dir',
        help='Output directory for CSV files (default: data)',
        default='data',
    )
    collect_parser.add_argument(
        '--env',
        help='Path to .env file',
        default=None,
    )
    collect_parser.add_argument(
        '--debug',
        action='store_true',
        help='Enable debug output for troubleshooting',
    )
    collect_parser.add_argument(
        '--live-inference',
        action='store_true',
        help='Enable live inference during collection (requires --model-dir)',
    )
    collect_parser.add_argument(
        '--model-dir',
        help='Path to trained model directory for live inference',
        default=None,
    )
    collect_parser.add_argument(
        '--window-size',
        type=int,
        help='Window size for live inference (default: 100)',
        default=100,
    )
    collect_parser.set_defaults(func=collect_command)

    # Plot command
    plot_parser = subparsers.add_parser(
        'plot',
        help='Live plot CSI amplitude data',
        description='Real-time visualization of CSI amplitude data from CSV files',
    )
    plot_parser.add_argument(
        'file',
        help='CSV file to plot (local path or user@host:/path for SSH)',
    )
    plot_parser.add_argument(
        '-s', '--subcarrier',
        type=int,
        default=10,
        help='Subcarrier index to plot (default: 10)',
    )
    plot_parser.add_argument(
        '-r', '--refresh',
        type=float,
        default=0.2,
        help='Refresh rate in seconds (default: 0.2)',
    )
    plot_parser.add_argument(
        '-m', '--maxpoints',
        type=int,
        default=20000,
        help='Maximum points to keep in memory (default: 20000)',
    )
    plot_parser.add_argument(
        '-l', '--limit',
        type=int,
        default=None,
        help='Limit display to last N points',
    )
    plot_parser.add_argument(
        '-w', '--window-size',
        type=int,
        default=10,
        help='Moving average window size (default: 10)',
    )
    plot_parser.add_argument(
        '--fs',
        type=float,
        default=None,
        help='Sampling rate in Hz (enables Butterworth filter)',
    )
    plot_parser.add_argument(
        '--fc',
        type=float,
        default=2.0,
        help='Cutoff frequency in Hz for Butterworth filter (default: 2.0)',
    )
    plot_parser.add_argument(
        '--order',
        type=int,
        default=4,
        help='Butterworth filter order (default: 4)',
    )
    plot_parser.set_defaults(func=plot_command)

    # Plot-data command
    plot_data_parser = subparsers.add_parser(
        'plot-data',
        help='Generate static plots from processed feature data',
        description='Generate various plots from feature CSV files',
    )
    plot_data_parser.add_argument(
        'input',
        nargs='?',  # Make optional for --list-plots
        help='Input feature CSV file',
    )
    plot_data_parser.add_argument(
        '--display',
        action='store_true',
        help='Display plots interactively (in addition to saving)',
    )
    plot_data_parser.add_argument(
        '-p', '--plots',
        help='Comma-separated list of plots to generate (default: all applicable)',
        default=None,
    )
    plot_data_parser.add_argument(
        '--list-plots',
        action='store_true',
        help='List available plot types and exit',
    )
    plot_data_parser.set_defaults(func=plot_data_command)

    # Process command
    process_parser = subparsers.add_parser(
        'process',
        help='Extract features from windowed CSI data',
        description='Convert raw CSI data to windowed features for machine learning',
    )
    process_parser.add_argument(
        'input',
        nargs='?',  # Make optional
        help='Input CSV file with raw CSI data',
    )
    process_parser.add_argument(
        'output',
        nargs='?',  # Make optional
        help='Output CSV file for extracted features',
    )
    process_parser.add_argument(
        '-w', '--window-size',
        type=int,
        default=100,
        help='Number of samples per window (default: 100)',
    )
    process_parser.add_argument(
        '-f', '--features',
        help='Comma-separated list of features to extract (default: all)',
        default=None,
    )
    process_parser.add_argument(
        '--list-features',
        action='store_true',
        help='List available features and exit',
    )
    process_parser.add_argument(
        '--labeled',
        action='store_true',
        help='Enable labeled mode: include labels in output and filter transition windows',
    )
    process_parser.add_argument(
        '--transition-buffer',
        type=int,
        default=1,
        help='Number of windows to discard before/after label transitions (default: 1)',
    )
    process_parser.add_argument(
        '--split',
        type=int,
        metavar='PERCENT',
        help='Split output by train/test percentage (e.g., --split 70 creates 70%% train, 30%% test). '
             'Split is stratified per label. Outputs: <output>-train.csv and <output>-test.csv',
    )
    process_parser.set_defaults(func=process_command)

    # Train command
    train_parser = subparsers.add_parser(
        'train',
        help='Train a machine learning model',
        description='Train an ML model on labeled CSI features for activity recognition',
    )
    train_parser.add_argument(
        'input',
        nargs='?',  # Make optional for --list-models
        help='Input CSV file with labeled features (from process --labeled)',
    )
    train_parser.add_argument(
        '-m', '--model',
        default='mlp',
        help='Model type to train (default: mlp)',
    )
    train_parser.add_argument(
        '-o', '--output-dir',
        help='Output directory for model (default: auto-generated with timestamp)',
        default=None,
    )
    train_parser.add_argument(
        '--train-split',
        type=float,
        default=0.7,
        help='Fraction of data for training (default: 0.7)',
    )
    train_parser.add_argument(
        '--val-split',
        type=float,
        default=0.15,
        help='Fraction of data for validation (default: 0.15)',
    )
    train_parser.add_argument(
        '--test-split',
        type=float,
        default=0.15,
        help='Fraction of data for testing (default: 0.15)',
    )
    train_parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducibility (default: 42)',
    )
    train_parser.add_argument(
        '--params',
        help='Model hyperparameters as key=value,key=value',
        default=None,
    )
    train_parser.add_argument(
        '--list-models',
        action='store_true',
        help='List available model types and exit',
    )
    train_parser.set_defaults(func=train_command)

    # Inference command
    inference_parser = subparsers.add_parser(
        'inference',
        help='Run model inference on new data',
        description='Generate predictions using a trained model (labels not required)',
    )
    inference_parser.add_argument(
        '--dataset',
        required=True,
        help='Input CSV file with features',
    )
    inference_parser.add_argument(
        '--model-dir',
        required=True,
        help='Directory containing trained model',
    )
    inference_parser.add_argument(
        '--output',
        help='Output CSV file for predictions (default: in model directory)',
        default=None,
    )
    inference_parser.add_argument(
        '--probabilities',
        action='store_true',
        help='Include class probabilities in output',
    )
    inference_parser.set_defaults(func=inference_command)

    # Evaluate command
    evaluate_parser = subparsers.add_parser(
        'evaluate',
        help='Evaluate model performance',
        description='Compute metrics on labeled test data',
    )
    evaluate_parser.add_argument(
        '--dataset',
        required=True,
        help='Input CSV file with labeled features',
    )
    evaluate_parser.add_argument(
        '--model-dir',
        required=True,
        help='Directory containing trained model',
    )
    evaluate_parser.add_argument(
        '--metrics',
        help='Comma-separated list of metrics to compute (default: all)',
        default=None,
    )
    evaluate_parser.add_argument(
        '--output-json',
        help='Output JSON file for metrics (default: in model directory)',
        default=None,
    )
    evaluate_parser.add_argument(
        '--output-txt',
        help='Output text file for report (default: in model directory)',
        default=None,
    )
    evaluate_parser.add_argument(
        '--list-metrics',
        action='store_true',
        help='List available metrics and exit',
    )
    evaluate_parser.set_defaults(func=evaluate_command)

    # Info command
    info_parser = subparsers.add_parser(
        'info',
        help='Show information about CSI Toolkit',
        description='Display information about CSI Toolkit modules and usage',
    )
    info_parser.set_defaults(func=info_command)

    # Parse arguments
    args = parser.parse_args()

    # Execute command
    return args.func(args)


if __name__ == '__main__':
    sys.exit(main())