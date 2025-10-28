#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import serial
import csv
import re
from datetime import datetime

# === Einstellungen ===
SERIAL_PORT = "/dev/cu.usbmodem1101"
BAUDRATE = 921600
CSV_FILENAME = f"csi_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"

# === CSV Header ===
CSV_HEADER = [
    "type", "seq", "mac", "rssi", "rate", "noise_floor",
    "fft_gain", "agc_gain", "channel", "local_timestamp",
    "sig_len", "rx_state", "len", "first_word", "data"
]

# === Regulärer Ausdruck, um CSI_DATA-Zeilen zu erkennen ===
CSI_PATTERN = re.compile(r'^CSI_DATA,(.+)$')

def parse_csi_line(line: str):
    if not line.startswith("CSI_DATA"):
        return None
    try:
        parts = list(csv.reader([line]))[0]
        if len(parts) < len(CSV_HEADER):
            return None
        return parts[:len(CSV_HEADER)]
    except Exception:
        return None

def main():
    print(f"Öffne {SERIAL_PORT} @ {BAUDRATE} Baud …")
    ser = serial.Serial(SERIAL_PORT, BAUDRATE, timeout=1)

    # start time
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
                    row_final = add_processed_fields(row)
                    if row:
                        writer.writerow(row_final)
                        f.flush()
                        elapsed_time = datetime.now() - start_time
                        rate = count / elapsed_time.total_seconds()
                        if elapsed_time.total_seconds() >= 1:
                            start_time = datetime.now()
                            count = 0
                        print(f"CSI frame {row[1]} saved (RSSI={row[3]} dBm), Rate={rate:.2f} Hz", end="\r")
        except KeyboardInterrupt:
            print("\nUser terminated the program.")
        finally:
            ser.close()
            print(f"File saved: {CSV_FILENAME}")
            
def add_processed_fields(row):
    """
    Hängt an die geparste Zeile die Subcarrier-Amplituden an.
    Erwartet:
      row[12] -> len  (Anzahl Rohwerte im data-Feld)
      row[14] -> data (JSON-Liste: [q0,i0,q1,i1,...] beim C6)
    Liefert:
      row + [amp_0, amp_1, ... amp_{N-1}]
    """
    if row is None:
        return None

    # Rohdaten holen
    data_str = row[14]
    # data_str ist typischerweise ein JSON-String wie "[0,1,-2,...]"
    # (ggf. mit Anführungszeichen vom CSV-Parser umschlossen)
    try:
        csi_raw = json.loads(data_str)
    except Exception:
        # Fallback: Anführungszeichen strippen und manuell parsen
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
                        # Ignoriere unlesbare Tokens
                        pass

    # Erwartete Länge prüfen (optional, robust gegen Inkonsistenzen)
    try:
        declared_len = int(row[12])
    except Exception:
        declared_len = None

    if declared_len is not None and declared_len != len(csi_raw):
        # Falls nötig, kann man hier loggen oder korrigieren
        pass

    # In I/Q-Paare umwandeln: beim ESP32-C6 sind die Werte interleaved (Q, I, Q, I, ...)
    n_pairs = len(csi_raw) // 2
    amplitudes = []
    for k in range(n_pairs):
        q = float(csi_raw[2 * k + 0])
        i = float(csi_raw[2 * k + 1])
        amp = (i * i + q * q) ** 0.5
        amplitudes.append(amp)

    # Amplituden am Ende anhängen
    return row + amplitudes


if __name__ == "__main__":
    main()