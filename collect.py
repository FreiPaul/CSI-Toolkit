#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import serial
import csv
import re
import json
import os
from datetime import datetime
from math import sqrt
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# === Settings ===
SERIAL_PORT = os.getenv("SERIAL_PORT", "/dev/cu.usbmodem1101")   # e.g., "COM3" on Windows
BAUDRATE = int(os.getenv("BAUDRATE", "921600"))
CSV_FILENAME = f"csi_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"

# === CSV Header ===
CSV_HEADER = [
    "type", "seq", "mac", "rssi", "rate", "noise_floor",
    "fft_gain", "agc_gain", "channel", "local_timestamp",
    "sig_len", "rx_state", "len", "first_word", "data", "amplitudes"
]

# === Regex pattern to detect CSI lines ===
CSI_PATTERN = re.compile(r'^CSI_DATA,(.+)$')

def parse_csi_line(line: str):
    """Safely parse a CSI_DATA line into individual CSV fields."""
    if not line.startswith("CSI_DATA"):
        return None
    try:
        parts = list(csv.reader([line]))[0]
        if len(parts) < len(CSV_HEADER) - 1:  # -1 because "amplitudes" will be added later
            return None
        return parts[:len(CSV_HEADER) - 1]
    except Exception:
        return None


def add_processed_fields(row):
    """
    Appends calculated amplitude information to the row.
    - Parses the 'data' JSON string (interleaved Q, I values).
    - Computes amplitude = sqrt(I^2 + Q^2) for each subcarrier.
    - Adds the amplitudes as a stringified list.
    """
    if not row:
        return None

    data_str = row[14]
    try:
        csi_raw = json.loads(data_str)
    except Exception:
        # fallback: manually parse if JSON decoding fails
        s = data_str.strip().strip('"').strip("'").strip()
        if s.startswith('[') and s.endswith(']'):
            s = s[1:-1]
        csi_raw = []
        if s:
            for tok in s.split(','):
                tok = tok.strip()
                if tok:
                    try:
                        csi_raw.append(int(tok))
                    except ValueError:
                        pass

    n_pairs = len(csi_raw) // 2
    amplitudes = []
    for k in range(n_pairs):
        q = float(csi_raw[2 * k + 0])
        i = float(csi_raw[2 * k + 1])
        amplitudes.append(round(sqrt(i * i + q * q), 4))

    # Add amplitude list as JSON string
    row_with_amp = row + [json.dumps(amplitudes)]
    return row_with_amp


def main():
    print(f"Opening {SERIAL_PORT} @ {BAUDRATE} baud …")
    ser = serial.Serial(SERIAL_PORT, BAUDRATE, timeout=1)

    start_time = datetime.now()

    with open(CSV_FILENAME, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(CSV_HEADER)

        count = 0
        print(f"Filename: {CSV_FILENAME}")

        try:
            while True:
                raw_line = ser.readline().decode(errors="ignore").strip()
                if not raw_line:
                    continue

                count += 1
                if CSI_PATTERN.match(raw_line):
                    row = parse_csi_line(raw_line)
                    if row:
                        row_final = add_processed_fields(row)
                        writer.writerow(row_final)
                        f.flush()
                        elapsed_time = datetime.now() - start_time
                        rate = count / max(elapsed_time.total_seconds(), 1e-6)
                        if elapsed_time.total_seconds() >= 1:
                            start_time = datetime.now()
                            count = 0
                        print(f"CSI frame {row[1]} saved (RSSI={row[3]} dBm), Rate={rate:.2f} Hz", end="\r")
        except KeyboardInterrupt:
            print("\nUser terminated the program.")
        finally:
            ser.close()
            print(f"File saved: {CSV_FILENAME}")


if __name__ == "__main__":
    main()