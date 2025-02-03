#!/usr/bin/env python3
"""
Upmix: filter_design.py

Provides approximate 4th-order Linkwitz-Riley FIR filter designs for high-pass or low-pass.
Also includes a helper to apply these FIR taps via `lfilter`.

Features:
1) `design_lr4_hp_fir` and `design_lr4_lp_fir`: 
   - Generate FIR taps for high-pass or low-pass filters, respectively.
   - If cutoff_hz <= 0, return a pass-through filter ([1.0]) so the signal is unchanged.
2) `apply_fir_filter`: 
   - Applies the FIR taps to an audio array using `scipy.signal.lfilter`.

Usage:
   - Import these functions into your processing scripts.
   - Example:
       taps_hp = design_lr4_hp_fir(sr=44100, cutoff_hz=180.0, numtaps=1025)
       output  = apply_fir_filter(input_wave, taps_hp)
"""

import numpy as np
from scipy.signal import firwin, lfilter

def design_lr4_hp_fir(sr: float,
                      cutoff_hz: float = 180.0,
                      numtaps: int = 1025) -> np.ndarray:
    """
    Approximate a 4th-order Linkwitz-Riley *high-pass* at `cutoff_hz`.
    If cutoff_hz <= 0, returns a pass-through filter [1.0].
    """
    if cutoff_hz <= 0:
        return np.array([1.0], dtype=np.float32)

    nyquist = 0.5 * sr
    norm_cutoff = cutoff_hz / nyquist
    taps = firwin(numtaps, norm_cutoff, pass_zero=False, window='hamming')
    return taps.astype(np.float32)

def design_lr4_lp_fir(sr: float,
                      cutoff_hz: float = 180.0,
                      numtaps: int = 1025) -> np.ndarray:
    """
    Approximate a 4th-order Linkwitz-Riley *low-pass* at `cutoff_hz`.
    If cutoff_hz <= 0, returns a pass-through filter [1.0].
    """
    if cutoff_hz <= 0:
        return np.array([1.0], dtype=np.float32)

    nyquist = 0.5 * sr
    norm_cutoff = cutoff_hz / nyquist
    taps = firwin(numtaps, norm_cutoff, pass_zero=True, window='hamming')
    return taps.astype(np.float32)

def apply_fir_filter(wave: np.ndarray, fir_taps: np.ndarray) -> np.ndarray:
    """
    Convolve `wave` with the FIR taps using `lfilter`.
    If `fir_taps` = [1.0], the signal passes through unmodified.
    """
    return lfilter(fir_taps, 1.0, wave)
