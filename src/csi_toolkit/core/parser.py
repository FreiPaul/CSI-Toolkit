"""Parsing utilities for CSI data."""

import json
import re
import csv
from io import StringIO
from typing import List, Optional, Dict, Any

from .constants import CSI_DATA_PREFIX
from .exceptions import ParsingError


def parse_csi_line(line: str) -> Optional[List[str]]:
    """
    Parse a CSI data line from serial input.

    Args:
        line: Raw line from serial port

    Returns:
        List of CSV fields if valid CSI_DATA line, None otherwise
    """
    line = line.strip()

    # Check for CSI_DATA prefix
    if not line.startswith(CSI_DATA_PREFIX):
        return None

    try:
        # Use proper CSV parsing to handle quoted fields containing commas
        reader = csv.reader(StringIO(line))
        csv_parts = next(reader)

        # Check we have enough fields (CSI_DATA + 14 data fields minimum)
        if len(csv_parts) < 15:
            return None

        # Return fields excluding CSI_DATA prefix
        # Fields are: seq, mac, rssi, rate, noise_floor, fft_gain, agc_gain,
        #            channel, local_timestamp, sig_len, rx_state, len, first_word, data
        return csv_parts[1:15]  # Return 14 fields after CSI_DATA
    except Exception as e:
        raise ParsingError(f"Failed to parse CSI line: {e}")


def parse_amplitude_json(json_str: str) -> List[float]:
    """
    Parse amplitude JSON string with fallback parsing.

    The JSON contains interleaved [Q, I, Q, I, ...] values that need
    to be parsed either as JSON or with fallback regex parsing.

    Args:
        json_str: JSON string containing amplitude data

    Returns:
        List of raw values (Q, I pairs)

    Raises:
        ParsingError: If parsing fails
    """
    if not json_str or json_str == "[]":
        return []

    try:
        # Try standard JSON parsing first
        amplitude_list = json.loads(json_str)
        if isinstance(amplitude_list, list):
            return amplitude_list
    except (json.JSONDecodeError, ValueError):
        # Fallback to regex parsing for malformed JSON
        try:
            # Extract numbers from string like "[1, 2, 3, ...]"
            matches = re.findall(r'-?\d+', json_str)
            if matches:
                return [int(m) for m in matches]
        except Exception as e:
            raise ParsingError(f"Fallback parsing failed: {e}")

    raise ParsingError(f"Could not parse amplitude JSON: {json_str[:100]}...")


def extract_amplitudes_from_row(row: Dict[str, Any]) -> Optional[List[float]]:
    """
    Extract amplitude values from a CSV row.

    Args:
        row: Dictionary containing CSV row data

    Returns:
        List of amplitude values or None if not present
    """
    amplitudes_str = row.get('amplitudes', '')

    if not amplitudes_str or amplitudes_str == '[]':
        return None

    try:
        return parse_amplitude_json(amplitudes_str)
    except ParsingError:
        # Log error but don't crash
        return None


def validate_csv_row(row: Dict[str, Any], required_fields: List[str]) -> bool:
    """
    Validate that a CSV row contains all required fields.

    Args:
        row: Dictionary containing CSV row data
        required_fields: List of required field names

    Returns:
        True if all required fields are present, False otherwise
    """
    return all(field in row for field in required_fields)