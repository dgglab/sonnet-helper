# sonnet_helper.py
# Sandesh Kalantre
# Some commonly used functions to load and analyze data from sonnet simulations

import numpy as np
import skrf as rf


def load_s21_touchstone(filepath, mag_db=True, phase_deg=True):
    """
    Load a Touchstone file and extract S21.

    Parameters
    ----------
    filepath : str
        Path to the Touchstone file (.ts)

    Returns
    -------
    freq_vec : ndarray
        Frequency vector in Hz
    s21_vec : ndarray
        Magnitude of S21 (in linear scale or dB based on mag_db)
    s21_phase_vec : ndarray
        Phase of S21 (in radians or degrees based on phase_deg)
    complex_s21_vec : ndarray
        Complex S21 values

    """
    ntwk = rf.Network(filepath, nports=2)

    freq_vec = ntwk.f  # Frequency in Hz
    complex_s21_vec = ntwk.s[:, 1, 0]  # S21 (port 2 <- port 1)
    s21_vec = np.abs(complex_s21_vec)
    if mag_db:
        s21_vec = 20 * np.log10(complex_s21_vec)

    s21_phase_vec = np.angle(complex_s21_vec)
    if phase_deg:
        s21_phase_vec = np.degrees(s21_phase_vec)

    return freq_vec, s21_vec, s21_phase_vec, complex_s21_vec


def load_sonnet_mdf(file_path):
    """
    Parses Sonnet MDF files by reading numbers directly into numpy. Assumes a 2 port network

    Returns a dictionary of datasets with the keys as params and data as a skrf Network object
    """
    datasets = {}
    current_params = {}

    # Buffers
    block_lines = []
    in_block = False

    with open(file_path, "r") as f:
        for line in f:
            clean_line = line.strip()

            # 1. Parse Parameters
            if clean_line.startswith("VAR"):
                parts = clean_line.replace("VAR", "").split("=")
                if len(parts) == 2:
                    p_name = parts[0].strip().replace('"', "")
                    p_val = float(parts[1].strip())
                    current_params[p_name] = p_val

            # 2. Start Block
            elif clean_line.startswith("BEGIN"):
                in_block = True
                block_lines = []

            # 3. End Block -> Process Data
            elif clean_line.startswith("END"):
                in_block = False
                if block_lines:
                    try:
                        # Join all lines and extract numbers
                        # Filter out comments (%) or headers (#)
                        valid_lines = [
                            l
                            for l in block_lines
                            if not (
                                l.startswith("%")
                                or l.startswith("#")
                                or l.startswith("!")
                                or l.startswith("REM")
                            )
                        ]

                        # Load data into a standard 2D numpy array
                        # Sonnet format: Freq  R11 I11  R12 I12  R21 I21  R22 I22
                        raw_data = np.loadtxt(valid_lines)

                        # Extract Frequency
                        freqs = raw_data[:, 0]

                        # Construct S-Matrix (N x 2 x 2)

                        n_points = len(freqs)
                        s_matrix = np.zeros((n_points, 2, 2), dtype=complex)

                        # S11
                        s_matrix[:, 0, 0] = raw_data[:, 1] + 1j * raw_data[:, 2]
                        # S12
                        s_matrix[:, 0, 1] = raw_data[:, 3] + 1j * raw_data[:, 4]
                        # S21
                        s_matrix[:, 1, 0] = raw_data[:, 5] + 1j * raw_data[:, 6]
                        # S22
                        s_matrix[:, 1, 1] = raw_data[:, 7] + 1j * raw_data[:, 8]

                        # Create Network from Arrays (No file I/O involved!)
                        ntwk = rf.Network()
                        ntwk.frequency = rf.Frequency.from_f(freqs, unit="hz")
                        ntwk.s = s_matrix
                        ntwk.z0 = 50  # Assume 50 ohm

                        # Store
                        param_key = tuple(sorted(current_params.items()))
                        ntwk.name = str(param_key)
                        datasets[param_key] = ntwk

                    except Exception as e:
                        print(f"Error parsing block {current_params}: {e}")

            # 4. Accumulate
            elif in_block:
                block_lines.append(clean_line)

    return datasets


def load_sonnet_mdf_mag_angle(file_path):
    """
    Parses Sonnet MDF files by reading numbers directly into numpy. Assumes a 2 port network

    Returns a dictionary of datasets with the keys as params and data as a skrf Network object
    """
    datasets = {}
    current_params = {}

    # Buffers
    block_lines = []
    in_block = False

    with open(file_path, "r") as f:
        for line in f:
            clean_line = line.strip()

            # 1. Parse Parameters
            if clean_line.startswith("VAR"):
                parts = clean_line.replace("VAR", "").split("=")
                if len(parts) == 2:
                    p_name = parts[0].strip().replace('"', "")
                    p_val = float(parts[1].strip())
                    current_params[p_name] = p_val

            # 2. Start Block
            elif clean_line.startswith("BEGIN"):
                in_block = True
                block_lines = []

            # 3. End Block -> Process Data
            elif clean_line.startswith("END"):
                in_block = False
                if block_lines:
                    try:
                        # Join all lines and extract numbers
                        # Filter out comments (%) or headers (#)
                        valid_lines = [
                            l
                            for l in block_lines
                            if not (
                                l.startswith("%")
                                or l.startswith("#")
                                or l.startswith("!")
                                or l.startswith("REM")
                            )
                        ]

                        # Load data into a standard 2D numpy array
                        # Sonnet format: Freq  R11 I11  R12 I12  R21 I21  R22 I22
                        raw_data = np.loadtxt(valid_lines)

                        # Extract Frequency
                        freqs = raw_data[:, 0]

                        # Construct S-Matrix (N x 2 x 2)

                        n_points = len(freqs)
                        s_matrix = np.zeros((n_points, 2, 2), dtype=complex)

                        # S11
                        s_matrix[:, 0, 0] = raw_data[:, 1] * np.e ** (
                            1j * raw_data[:, 2]
                        )
                        # S12
                        s_matrix[:, 0, 1] = raw_data[:, 3] * np.e ** (
                            1j * raw_data[:, 4]
                        )
                        # S21
                        s_matrix[:, 1, 0] = raw_data[:, 5] * np.e ** (
                            1j * raw_data[:, 6]
                        )
                        # S22
                        s_matrix[:, 1, 1] = raw_data[:, 7] * np.e ** (
                            1j * raw_data[:, 8]
                        )

                        # Create Network from Arrays (No file I/O involved!)
                        ntwk = rf.Network()
                        ntwk.frequency = rf.Frequency.from_f(freqs, unit="hz")
                        ntwk.s = s_matrix
                        ntwk.z0 = 50  # Assume 50 ohm

                        # Store
                        param_key = tuple(sorted(current_params.items()))
                        ntwk.name = str(param_key)
                        datasets[param_key] = ntwk

                    except Exception as e:
                        print(f"Error parsing block {current_params}: {e}")

            # 4. Accumulate
            elif in_block:
                block_lines.append(clean_line)

    return datasets
