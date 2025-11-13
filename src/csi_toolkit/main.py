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

    # Create and start collector
    collector = SerialCollector(config)
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
        plotter.start()
    except KeyboardInterrupt:
        print("\nPlotting stopped by user")
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    return 0


def process_command(args):
    """Handle the process subcommand."""
    print("Data processing functionality will be implemented in future versions")
    print("Available processing modules:")
    print("  - Amplitude calculation")
    print("  - Feature extraction")
    print("  - Signal filtering")
    return 0


def info_command(args):
    """Handle the info subcommand."""
    print("CSI Toolkit - Channel State Information Processing Pipeline")
    print("=" * 60)
    print("\nModules:")
    print("  - Collection: Serial data acquisition from ESP32")
    print("  - Visualization: Real-time plotting with filtering")
    print("  - Processing: Signal processing and feature extraction")
    print("  - I/O: CSV reading/writing with SSH support")
    print("\nData Flow:")
    print("  ESP32 -> Serial -> CSV -> Processing -> Visualization")
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

    # Process command (placeholder)
    process_parser = subparsers.add_parser(
        'process',
        help='Process CSI data (batch processing)',
        description='Batch processing of CSI data files',
    )
    process_parser.set_defaults(func=process_command)

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