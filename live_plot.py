#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import csv
import json
import threading
import time
from collections import deque
from math import sqrt
from pathlib import Path

import matplotlib.pyplot as plt

# Optional SciPy for Butterworth filtering
try:
    from scipy.signal import butter, filtfilt
    HAS_SCIPY = True
except Exception:
    HAS_SCIPY = False

# python live_plot.py csi_log_20251028_120402.csv --refresh 0.2 --subcarrier 10
def parse_args():
    p = argparse.ArgumentParser(
        description="Live-tail a CSI CSV and plot amplitudes with low-pass filtering."
    )
    p.add_argument("csvfile", type=str, help="Path to the growing CSI CSV file")
    p.add_argument("--refresh", type=float, default=0.2,
                   help="Graph refresh interval in seconds (default: 0.2)")
    p.add_argument("--subcarrier", type=int, default=10,
                   help="1-based subcarrier index to plot (default: 10)")
    p.add_argument("--fs", type=float, default=None,
                   help="Packet rate in Hz. If set AND SciPy is available, use Butterworth low-pass.")
    p.add_argument("--fc", type=float, default=2.0,
                   help="Low-pass cutoff in Hz for Butterworth (requires --fs). Default: 2.0")
    p.add_argument("--order", type=int, default=4,
                   help="Butterworth filter order (default: 4)")
    p.add_argument("--ma", type=int, default=10,
                   help="Moving-average window (packets) when SciPy/--fs not used. Default: 10")
    p.add_argument("--maxpoints", type=int, default=20000,
                   help="Max packets to keep in memory for plotting (ring buffer). Default: 20000")
    p.add_argument("--limit", type=int, default=None,
                   help="Limit of points to plot (rolling window). If set, only the last N points are displayed.")
    return p.parse_args()


def moving_average(z, N):
    """Zero-phase-ish smoothing via forward/backward box filter (filtfilt equivalent for MA)."""
    if len(z) == 0:
        return z
    if N <= 1:
        return z[:]
    # Causal MA
    y = []
    s = 0.0
    q = deque()
    for val in z:
        q.append(val)
        s += val
        if len(q) > N:
            s -= q.popleft()
        y.append(s / len(q))
    # Reverse and repeat for zero-phase effect
    y2 = []
    s = 0.0
    q.clear()
    for val in reversed(y):
        q.append(val)
        s += val
        if len(q) > N:
            s -= q.popleft()
        y2.append(s / len(q))
    y2.reverse()
    return y2


def lowpass(series, fs=None, fc=2.0, order=4, maN=10):
    """Apply Butterworth (if fs & SciPy available) else moving-average."""
    if fs is not None and HAS_SCIPY:
        # Protect against bad cutoff
        if fc <= 0 or fs <= 0 or fc >= fs / 2:
            # Fallback to MA
            return moving_average(series, maN)
        from numpy import array
        w = fc / (fs * 0.5)
        b, a = butter(order, w, btype="low")
        try:
            return filtfilt(b, a, array(series)).tolist()
        except Exception:
            return moving_average(series, maN)
    else:
        return moving_average(series, maN)


def tail_csv(
    path: Path,
    on_row,
    stop_event: threading.Event,
    maxpoints: int = 20000,
    pkt_counter_ref: list = None,
):
    """
    Tail the CSV file and call on_row(row_dict) for each new parsed row.
    Expects an 'amplitudes' column containing a JSON array string.
    pkt_counter_ref: A list containing the packet counter to update (for tracking skipped rows)
    """
    # Wait until file exists and has at least a header
    while not stop_event.is_set():
        if path.exists() and path.stat().st_size > 0:
            break
        time.sleep(0.2)

    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        # Read header
        header = None
        while not stop_event.is_set():
            pos = f.tell()
            line = f.readline()
            if not line:
                time.sleep(0.1)
                f.seek(pos)
                continue
            row = next(csv.reader([line]))
            header = row
            break

        if header is None:
            return

        # Resolve important columns
        # We require at least "amplitudes"; others are optional for printing
        try:
            idx_amp = header.index("amplitudes")
        except ValueError:
            raise RuntimeError('CSV must include an "amplitudes" column.')

        idx_seq = header.index("seq") if "seq" in header else None
        idx_rssi = header.index("rssi") if "rssi" in header else None

        # Consume the rest (including already-present rows)
        # First, collect all existing rows to handle large files efficiently
        all_lines = []
        for line in f:
            if line.strip():  # Skip empty lines
                all_lines.append(line)

        # Only process the last maxpoints rows if file is large
        total_rows = len(all_lines)
        skipped_rows = max(0, total_rows - maxpoints)
        lines_to_process = all_lines[-maxpoints:] if total_rows > maxpoints else all_lines

        # Update the packet counter to account for skipped rows
        if pkt_counter_ref is not None and len(pkt_counter_ref) > 0:
            pkt_counter_ref[0] = skipped_rows

        for line in lines_to_process:
            try:
                row = next(csv.reader([line]))
            except Exception:
                continue
            if len(row) <= idx_amp:
                continue
            on_row(row, header, idx_amp, idx_seq, idx_rssi)

        # Now live tail
        while not stop_event.is_set():
            pos = f.tell()
            line = f.readline()
            if not line:
                time.sleep(0.05)
                f.seek(pos)
                continue
            try:
                row = next(csv.reader([line]))
            except Exception:
                continue
            if len(row) <= idx_amp:
                continue
            on_row(row, header, idx_amp, idx_seq, idx_rssi)


def main():
    args = parse_args()
    csv_path = Path(args.csvfile)

    # Shared buffers (append-only)
    pkt_idx = []         # packet index (1..N)
    mean_amp = []        # per-packet mean amplitude
    sc_amp = []          # per-packet single-subcarrier amplitude
    sc_index_1based = max(1, args.subcarrier)
    pkt_counter_ref = [0]  # total packets processed (in list to share with thread)

    lock = threading.Lock()
    stop_event = threading.Event()

    def on_row(row, header, idx_amp, idx_seq, idx_rssi):
        nonlocal sc_index_1based
        try:
            amp_list = json.loads(row[idx_amp])
        except Exception:
            # try manual parse if malformed
            s = row[idx_amp].strip().strip('"').strip("'")
            if s.startswith("[") and s.endswith("]"):
                s = s[1:-1]
            amp_list = []
            for tok in s.split(","):
                tok = tok.strip()
                if tok:
                    try:
                        amp_list.append(float(tok))
                    except ValueError:
                        pass

        if len(amp_list) == 0:
            return

        # Clamp subcarrier index (1-based)
        if sc_index_1based > len(amp_list):
            sc_index_1based = len(amp_list)

        with lock:
            pkt_counter_ref[0] += 1
            pkt_idx.append(pkt_counter_ref[0])
            mean_amp.append(sum(amp_list) / len(amp_list))
            sc_amp.append(amp_list[sc_index_1based - 1])

            # Limit memory
            if len(pkt_idx) > args.maxpoints:
                drop = len(pkt_idx) - args.maxpoints
                del pkt_idx[:drop]
                del mean_amp[:drop]
                del sc_amp[:drop]

    # Start tail thread
    t = threading.Thread(
        target=tail_csv, args=(csv_path, on_row, stop_event, args.maxpoints, pkt_counter_ref), daemon=True
    )
    t.start()

    # Matplotlib live plot
    plt.figure("CSI amplitudes (live)", figsize=(10, 7))
    ax1 = plt.subplot(2, 1, 1)
    ax2 = plt.subplot(2, 1, 2)
    ln1_raw, = ax1.plot([], [], label="mean amplitude (raw)")
    ln1_flt, = ax1.plot([], [], linewidth=1.8, label="mean amplitude (low-pass)")
    ax1.set_xlabel("Packet")
    ax1.set_ylabel("Amplitude (a.u.)")
    ax1.set_title("Mean amplitude across subcarriers")
    ax1.grid(True)
    ax1.legend(loc="best")

    ln2_raw, = ax2.plot([], [], label=f"SC amplitude (raw)")
    ln2_flt, = ax2.plot([], [], linewidth=1.8, label=f"SC amplitude (low-pass)")
    ax2.set_xlabel("Packet")
    ax2.set_ylabel("Amplitude (a.u.)")
    ax2.set_title(f"Single subcarrier amplitude (SC={args.subcarrier})")
    ax2.grid(True)
    ax2.legend(loc="best")

    last_len = 0
    try:
        while True:
            time.sleep(args.refresh)
            with lock:
                n = len(pkt_idx)
                if n == 0:
                    continue
                x = pkt_idx[:]
                y1 = mean_amp[:]
                y2 = sc_amp[:]

            # Apply rolling window limit if specified
            if args.limit is not None and n > args.limit:
                x = x[-args.limit:]
                y1 = y1[-args.limit:]
                y2 = y2[-args.limit:]

            # Filtering
            y1_f = lowpass(y1, fs=args.fs, fc=args.fc, order=args.order, maN=args.ma)
            y2_f = lowpass(y2, fs=args.fs, fc=args.fc, order=args.order, maN=args.ma)

            # Update plots
            ln1_raw.set_data(x, y1)
            ln1_flt.set_data(x, y1_f)
            ln2_raw.set_data(x, y2)
            ln2_flt.set_data(x, y2_f)

            # Rescale axes on growth
            for ax, yy in ((ax1, y1 + y1_f), (ax2, y2 + y2_f)):
                ax.relim()
                ax.autoscale_view()

            # Update xlim whenever data changes (not just when count changes)
            if len(x) > 1:
                ax1.set_xlim(x[0], x[-1])
                ax2.set_xlim(x[0], x[-1])

            last_len = n

            plt.pause(0.001)
            # Keep checking window events responsive
            if not plt.fignum_exists(plt.gcf().number):
                break

    except KeyboardInterrupt:
        pass
    finally:
        stop_event.set()
        t.join(timeout=1.0)


if __name__ == "__main__":
    main()