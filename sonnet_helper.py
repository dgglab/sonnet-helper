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
