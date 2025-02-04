#!/usr/bin/env python3
"""
Multi‐Band STFT Upmix System with WOLA Synthesis

This module implements a multi‐band STFT‐based upmix system for stereo audio.
It provides functionality to:
  • Compute various window functions (including Blackman–Harris, Hann, etc.)
  • Design a synthesis window using the Weighted Overlap–Add (WOLA) technique
    given an analysis window (e.g. Blackman–Harris) at a specified overlap
    (e.g. 75%).
  • Perform forward and inverse STFT operations.
  • Automatically compute suitable block sizes for each frequency band based on
    the band’s lower frequency limit.
  • Apply frequency‐domain band limiting using either hard zeroing or a raised‐cosine filter.
  • Extract center, left, and right components from stereo signals per band and sum them.
  • Process bands in parallel using a thread pool.
  • Visualize the analysis and synthesis windows and their overlap–add behavior.

Usage:
  - Run this module directly to process an input stereo file (default: "in/eyes.wav")
    and output a processed stereo file (in "out/").
  - Alternatively, import functions and classes for custom processing.
"""

import os
import math
import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt
from typing import Callable, List
from concurrent.futures import ThreadPoolExecutor

###############################################################################
# Constants
###############################################################################
EPS = 1e-12

###############################################################################
# Window Functions
###############################################################################
# -- Primary window function used in the new WOLA system
def make_blackman_harris(N: int) -> np.ndarray:
    """
    Generate a Blackman–Harris window of length N.
    Coefficients are chosen for low sidelobes.
    """
    n = np.arange(N)
    a0, a1, a2, a3 = 0.35875, 0.48829, 0.14128, 0.01168
    w = (a0
         - a1 * np.cos(2 * np.pi * n / (N - 1))
         + a2 * np.cos(4 * np.pi * n / (N - 1))
         - a3 * np.cos(6 * np.pi * n / (N - 1)))
    return w.astype(np.float32)

# -- Additional window functions (retained for prototyping and alternative uses)
def make_sqrt_hann(N: int) -> np.ndarray:
    """Generate a square-root Hann window, commonly used for 50% overlap."""
    h = np.hanning(N)
    return np.sqrt(h).astype(np.float32)

def make_hann(N: int) -> np.ndarray:
    """Generate a standard Hann window (often used with 75% overlap)."""
    return np.hanning(N).astype(np.float32)

def make_blackman(N: int) -> np.ndarray:
    """Generate a Blackman window."""
    return np.blackman(N).astype(np.float32)

def make_hamming(N: int) -> np.ndarray:
    """Generate a Hamming window."""
    return np.hamming(N).astype(np.float32)

def make_rect(N: int) -> np.ndarray:
    """Generate a rectangular (boxcar) window."""
    return np.ones(N, dtype=np.float32)

###############################################################################
# WOLA Synthesis-Window Design
###############################################################################
def design_wola_synthesis_window(analysis_window: np.ndarray, overlap: float) -> np.ndarray:
    """
    Design a synthesis window for WOLA reconstruction.
    
    Given an analysis window w_A(n), the synthesis window is computed as:
      w_S(n) = w_A(n) / (sum_{k} [w_A^2(n + k*H)] + EPS)
    where H = hop size = L * (1 - overlap) and the summation is over all overlapping frames.
    For example, with 75% overlap (overlap=0.75, H=0.25L), 4 windows overlap.
    """
    L = len(analysis_window)
    hop = int(L * (1.0 - overlap))
    if hop < 1:
        raise ValueError("Overlap too large; resulting hop size < 1.")
    
    K = int(round(1.0 / (1.0 - overlap)))  # e.g., 4 for 75% overlap
    syn_window = np.zeros(L, dtype=analysis_window.dtype)
    
    # For each sample in the window, compute the sum over overlapping analysis windows
    for n in range(L):
        sum_sq = 0.0
        for k in range(K):
            idx = (n + k * hop) % L  # simulate infinite tiling via modulo
            sum_sq += analysis_window[idx] ** 2
        syn_window[n] = analysis_window[n] / (sum_sq + EPS)
    
    return syn_window

###############################################################################
# STFT Helpers
###############################################################################
def forward_stft(block: np.ndarray, analysis_win: np.ndarray) -> np.ndarray:
    """
    Compute the forward Short-Time Fourier Transform (STFT) for a block.
    
    Parameters:
      block       : The time-domain signal block.
      analysis_win: The analysis window to apply.
      
    Returns:
      The rFFT of the windowed block.
    """
    windowed = block * analysis_win
    return np.fft.rfft(windowed)

def inverse_stft(spec: np.ndarray, synthesis_win: np.ndarray) -> np.ndarray:
    """
    Compute the inverse Short-Time Fourier Transform (iSTFT) for a spectrum.
    
    Parameters:
      spec         : The frequency-domain spectrum.
      synthesis_win: The synthesis window to apply.
      
    Returns:
      The time-domain signal (as float32), weighted by the synthesis window.
    """
    rec = np.fft.irfft(spec).astype(np.float32)
    rec *= synthesis_win
    return rec

###############################################################################
# Miscellaneous Utilities
###############################################################################
def freq_to_bin(freq_hz: float, sr: float, fft_size: int) -> int:
    """
    Convert a frequency in Hz to the corresponding FFT bin index (for rFFT).
    
    Parameters:
      freq_hz : Frequency in Hertz.
      sr      : Sampling rate.
      fft_size: FFT size (number of time-domain samples).
      
    Returns:
      The nearest bin index corresponding to freq_hz.
    """
    return int(round(freq_hz / (sr / float(fft_size))))

def next_power_of_2(x: int) -> int:
    """
    Compute the smallest power of 2 greater than or equal to x.
    
    Parameters:
      x: An integer.
      
    Returns:
      The smallest power of 2 that is >= x.
    """
    if x < 1:
        return 1
    power = 1
    while power < x:
        power <<= 1
    return power

def compute_block_size_for_low_freq(f_low: float, sr: float, max_block_size: int = 2**16, threshold_factor: float = 32) -> int:
    """
    Compute an appropriate block size for STFT based on the lower frequency bound.
    
    The block size is chosen so that a certain number of cycles of the low frequency
    are captured. Here, we use:
         threshold = (sr * threshold_factor) / f_low
    and then round up to the next power of 2, clamping to max_block_size.
    
    (Note: An earlier version used sr*20 for the threshold; adjust the factor as needed.)
    
    Parameters:
      f_low           : Lower frequency bound for the band.
      sr              : Sampling rate.
      max_block_size  : Maximum allowed block size.
      threshold_factor: Factor multiplied by the sample rate (default is 32).
      
    Returns:
      The computed block size.
    """
    if f_low <= 0.0:
        return max_block_size
    threshold = (sr * threshold_factor) / f_low
    candidate_blk = next_power_of_2(int(np.ceil(threshold)))
    return min(candidate_blk, max_block_size)


def hp_freq_to_crossover_width(hp_freq: float) -> float:
    """
    Compute the crossover width (fade zone) in Hz given a high-pass frequency.
    
    By default, the crossover width is set to 25% of the high-pass frequency.
    
    Parameters:
      hp_freq: The high-pass frequency in Hz.
      
    Returns:
      The crossover width in Hz.
    """
    return hp_freq * 0.25

###############################################################################
# Multi-Band Extractor with WOLA
###############################################################################
class MultiBandExtractorAccu:
    """
    Per-band STFT extractor that:
      • Limits (or fades) frequencies outside [f_low, f_high] using either a
        hard-zero or raised-cosine approach.
      • Computes left/center/right (L/C/R) signals via cross-spectral analysis.
      • Overlap–adds successive blocks for continuous output.
      
    The WOLA design is applied by using a separate synthesis window computed from
    the analysis window.
    
    Attributes:
      block_size         : STFT block length.
      overlap            : Fractional overlap between successive blocks.
      hop_size           : Number of new samples per block (block_size * (1-overlap)).
      analysis_window    : Window for forward STFT (e.g., Blackman–Harris).
      synthesis_window   : Window for inverse STFT (computed via WOLA).
      sr                 : Sampling rate.
      f_low, f_high      : Lower and upper frequency limits for the band.
      xover_mode         : 'hard_zero' or 'raised_cosine' for frequency limiting.
      xover_width_low_hz : Fade width (Hz) near the lower bound.
      xover_width_high_hz: Fade width (Hz) near the upper bound.
    """
    def __init__(self,
                 block_size: int,
                 overlap: float,
                 window_func: Callable[[int], np.ndarray],
                 f_low: float,
                 f_high: float,
                 sr: float,
                 xover_mode: str = "hard_zero",
                 xover_width_low_hz: float = 50.0,
                 xover_width_high_hz: float = 50.0):
        self.block_size = block_size
        self.overlap    = overlap
        self.hop_size   = int(block_size * (1 - overlap))
        if self.hop_size < 1:
            raise ValueError("Overlap too large; hop size < 1 is not allowed.")

        # Define separate analysis and synthesis windows (WOLA)
        self.analysis_window  = window_func(block_size)
        self.synthesis_window = design_wola_synthesis_window(self.analysis_window, overlap)

        self.sr     = sr
        self.f_low  = f_low
        self.f_high = f_high

        self.xover_mode          = xover_mode
        self.xover_width_low_hz  = xover_width_low_hz
        self.xover_width_high_hz = xover_width_high_hz

        # Overlap-add accumulators for center, left, and right signals
        self.accumC = np.zeros(block_size, dtype=np.float32)
        self.accumL = np.zeros(block_size, dtype=np.float32)
        self.accumR = np.zeros(block_size, dtype=np.float32)

    def _hard_zero_filter(self, specL: np.ndarray, specR: np.ndarray, bin_low: int, bin_high: int):
        """
        Zero out frequency bins outside the passband [bin_low, bin_high].
        """
        specL[:bin_low] = 0
        specR[:bin_low] = 0
        specL[bin_high+1:] = 0
        specR[bin_high+1:] = 0

    def _raised_cosine_filter(self, specL: np.ndarray, specR: np.ndarray, bin_low: int, bin_high: int, fft_size: int):
        """
        Apply a raised-cosine (half-cosine) fade at the band edges.
        This smoothly fades frequencies below bin_low and above bin_high.
        """
        n_bins = len(specL)
        # Ensure proper ordering and bounds
        if bin_low > bin_high:
            bin_low, bin_high = bin_high, bin_low
        bin_low  = max(bin_low, 0)
        bin_high = min(bin_high, n_bins - 1)

        if bin_low > bin_high:
            specL[:] = 0
            specR[:] = 0
            return

        # Convert fade widths (in Hz) to bin counts
        fade_bins_low  = freq_to_bin(self.xover_width_low_hz,  self.sr, fft_size)
        fade_bins_high = freq_to_bin(self.xover_width_high_hz, self.sr, fft_size)

        # Fade in near the low-frequency bound (if applicable)
        if self.f_low > 0:
            fade_in_start = max(0, bin_low - fade_bins_low)
            specL[:fade_in_start] = 0
            specR[:fade_in_start] = 0
            if fade_in_start < bin_low:
                fade_in_len = bin_low - fade_in_start
                for i in range(fade_in_len):
                    idx = fade_in_start + i
                    x = (i + 0.5) / fade_in_len
                    alpha = 0.5 * (1.0 - np.cos(np.pi * x))  # ramps from 0 to 1
                    specL[idx] *= alpha
                    specR[idx] *= alpha

        # Fade out near the high-frequency bound (if applicable)
        if self.f_high < self.sr * 0.5:
            fade_out_start = bin_high + 1
            fade_out_end   = fade_out_start + fade_bins_high
            if fade_out_start < n_bins:
                fade_out_end = min(fade_out_end, n_bins)
                fade_out_len = fade_out_end - fade_out_start
                for i in range(fade_out_len):
                    idx = fade_out_start + i
                    x = (i + 0.5) / fade_out_len
                    alpha = 0.5 * (1.0 + np.cos(np.pi * x))  # ramps from 1 to 0
                    specL[idx] *= alpha
                    specR[idx] *= alpha
                if fade_out_end < n_bins:
                    specL[fade_out_end:] = 0
                    specR[fade_out_end:] = 0

    def _band_limit(self, specL: np.ndarray, specR: np.ndarray):
        """
        Limit or fade frequency bins outside the desired passband [f_low, f_high].
        """
        n_bins   = len(specL)
        fft_size = (n_bins - 1) * 2
        bin_low  = freq_to_bin(self.f_low,  self.sr, fft_size)
        bin_high = freq_to_bin(self.f_high, self.sr, fft_size)
        if bin_low > bin_high:
            bin_low, bin_high = bin_high, bin_low

        if self.xover_mode == "hard_zero":
            self._hard_zero_filter(specL, specR, bin_low, bin_high)
        elif self.xover_mode == "raised_cosine":
            self._raised_cosine_filter(specL, specR, bin_low, bin_high, fft_size)
        else:
            # Default to hard zero if an unknown mode is specified
            self._hard_zero_filter(specL, specR, bin_low, bin_high)

    def process_stereo_chunk(self, blkL: np.ndarray, blkR: np.ndarray) -> tuple:
        """
        Process a stereo block:
          1. Compute the forward STFT (using the analysis window).
          2. Limit/fade frequency bins outside [f_low, f_high].
          3. Compute cross-spectral parameters to extract center and side (left/right) signals.
          4. Compute the inverse STFT (using the synthesis window) for each component.
          5. Overlap-add the reconstructed signals.
          
        Returns:
          A tuple (center_chunk, left_chunk, right_chunk) corresponding to the first hop_size samples.
        """
        # Forward STFT on each channel
        specL = forward_stft(blkL, self.analysis_window)
        specR = forward_stft(blkR, self.analysis_window)

        # Apply frequency-domain band limiting
        self._band_limit(specL, specR)

        # Cross-spectral analysis for L/C/R extraction
        cross     = specL * np.conjugate(specR)
        cross_mag = np.abs(cross)
        magL      = np.abs(specL)
        magR      = np.abs(specR)
        denom     = (magL * magR) + EPS
        coherence = cross_mag / denom
        balance   = (magL - magR) / (magL + magR + EPS)
        centerFactor = coherence * (1.0 - np.abs(balance))

        spec_center = 0.5 * centerFactor * (specL + specR)
        spec_left   = specL - spec_center
        spec_right  = specR - spec_center

        # Inverse STFT using the synthesis window
        rec_center = inverse_stft(spec_center, self.synthesis_window)
        rec_left   = inverse_stft(spec_left,   self.synthesis_window)
        rec_right  = inverse_stft(spec_right,  self.synthesis_window)

        # Overlap–add accumulation
        self.accumC += rec_center
        self.accumL += rec_left
        self.accumR += rec_right

        # Extract the output chunk (first hop_size samples)
        out_c = self.accumC[:self.hop_size].copy()
        out_l = self.accumL[:self.hop_size].copy()
        out_r = self.accumR[:self.hop_size].copy()

        # Shift accumulators for the next block
        self.accumC[:-self.hop_size] = self.accumC[self.hop_size:]
        self.accumL[:-self.hop_size] = self.accumL[self.hop_size:]
        self.accumR[:-self.hop_size] = self.accumR[self.hop_size:]
        self.accumC[-self.hop_size:] = 0
        self.accumL[-self.hop_size:] = 0
        self.accumR[-self.hop_size:] = 0

        return out_c, out_l, out_r

    def flush_final(self) -> tuple:
        """
        Flush any remaining samples in the overlap–add accumulators.
        
        Returns:
          A tuple (leftover_center, leftover_left, leftover_right).
        """
        leftover_c = self.accumC.copy()
        leftover_l = self.accumL.copy()
        leftover_r = self.accumR.copy()
        self.accumC[:] = 0
        self.accumL[:] = 0
        self.accumR[:] = 0
        return leftover_c, leftover_l, leftover_r

    def process_all_blocks(self, L: np.ndarray, R: np.ndarray) -> tuple:
        """
        Process the entire input signals (L and R channels) for this frequency band.
        
        The method pads the signals as needed so that every block is of length block_size,
        processes each block sequentially, and finally flushes any remaining samples.
        
        Returns:
          Tuple of time-domain signals (center, left, right) for this band,
          trimmed to the original signal length.
        """
        N = len(L)
        leftover_len = self.block_size - self.hop_size
        needed       = leftover_len
        num_hops     = math.ceil((N - needed) / self.hop_size)
        padded_len   = num_hops * self.hop_size + needed
        pad_amt      = max(0, padded_len - N)

        L_pad = np.pad(L, (0, pad_amt), mode='constant')
        R_pad = np.pad(R, (0, pad_amt), mode='constant')

        out_c, out_l, out_r = [], [], []
        idx = 0
        while idx < len(L_pad):
            blkL = L_pad[idx : idx + self.block_size]
            blkR = R_pad[idx : idx + self.block_size]
            if len(blkL) < self.block_size:
                # Zero-pad the last block if it is incomplete
                blkL = np.pad(blkL, (0, self.block_size - len(blkL)), mode='constant')
                blkR = np.pad(blkR, (0, self.block_size - len(blkR)), mode='constant')
            c_chunk, l_chunk, r_chunk = self.process_stereo_chunk(blkL, blkR)
            out_c.append(c_chunk)
            out_l.append(l_chunk)
            out_r.append(r_chunk)
            idx += self.hop_size

        # Flush any remaining overlap–added samples
        lc, ll, lr = self.flush_final()
        out_c.append(lc)
        out_l.append(ll)
        out_r.append(lr)

        final_c = np.concatenate(out_c)[:N]
        final_l = np.concatenate(out_l)[:N]
        final_r = np.concatenate(out_r)[:N]

        return final_c, final_l, final_r

###############################################################################
# Parallel Multi-Band Extraction
###############################################################################
def extract_center_left_right_multi_band_in_memory(
    L: np.ndarray,
    R: np.ndarray,
    sr: float,
    band_extractors: List[MultiBandExtractorAccu],
) -> tuple:
    """
    Process the input stereo signals through multiple frequency bands in parallel.
    
    Each band extractor processes the full input signal and returns its band-limited
    center, left, and right outputs. The final output is obtained by summing the
    contributions from all bands.
    
    Parameters:
      L, R          : Left and right channel signals.
      sr            : Sampling rate.
      band_extractors: List of MultiBandExtractorAccu instances (one per band).
    
    Returns:
      A tuple (final_center, final_left, final_right) of the summed outputs.
    """
    results = []
    with ThreadPoolExecutor() as executor:
        futures = [executor.submit(bex.process_all_blocks, L, R) for bex in band_extractors]
        results = [f.result() for f in futures]

    N = len(L)
    final_center = np.zeros(N, dtype=np.float32)
    final_left   = np.zeros(N, dtype=np.float32)
    final_right  = np.zeros(N, dtype=np.float32)

    for (c_band, l_band, r_band) in results:
        final_center += c_band
        final_left   += l_band
        final_right  += r_band

    return final_center, final_left, final_right

###############################################################################
# Chain Bands
###############################################################################
def chain_bands(
    band_edges: List[float],
    overlap: float,
    window_func: Callable[[int], np.ndarray],
    sr: float,
    xover_mode: str = "raised_cosine",
) -> List[MultiBandExtractorAccu]:
    """
    Divide the frequency spectrum into consecutive bands defined by band_edges.
    
    For each band [f_low, f_high], this function:
      - Computes a suitable block size based on f_low.
      - Determines the crossover (fade) widths.
      - Instantiates a MultiBandExtractorAccu with the chosen parameters.
    
    If the last band edge is below sr/2, sr/2 is appended as the upper bound.
    
    Parameters:
      band_edges : List of frequency edges (in Hz) defining band boundaries.
      overlap    : Fractional overlap for STFT blocks (e.g., 0.75 for 75% overlap).
      window_func: Function to generate the analysis window.
      sr         : Sampling rate.
      xover_mode : Crossover mode ('hard_zero' or 'raised_cosine').
    
    Returns:
      A list of MultiBandExtractorAccu objects (one per band).
    """
    # Ensure the top band extends to Nyquist if not already specified
    if band_edges[-1] < (sr / 2.0):
        band_edges = list(band_edges) + [sr / 2.0]

    extractors = []
    prev_xover_high = 0.0

    for i in range(len(band_edges) - 1):
        f_low  = band_edges[i]
        f_high = band_edges[i + 1]
        block_size = compute_block_size_for_low_freq(f_low, sr)
        
        # Set crossover widths: previous band’s high fade and current band’s low fade.
        xover_low  = prev_xover_high
        xover_high = hp_freq_to_crossover_width(f_high)
        print(
            f"[Band {i+1}] f_low={f_low:.1f} Hz, "
            f"f_high={f_high:.1f} Hz, block_size={block_size}, "
            f"xover_low={xover_low:.1f} Hz, xover_high={xover_high:.1f} Hz"
        )

        ext = MultiBandExtractorAccu(
            block_size         = block_size,
            overlap            = overlap,
            window_func        = window_func,
            f_low              = f_low,
            f_high             = f_high,
            sr                 = sr,
            xover_mode         = xover_mode,
            xover_width_low_hz = xover_low,
            xover_width_high_hz= xover_high
        )
        extractors.append(ext)
        prev_xover_high = xover_high

    return extractors

###############################################################################
# Visualization Helper
###############################################################################
def visualize_windows(analysis_window: np.ndarray,
                      synthesis_window: np.ndarray,
                      overlap: float):
    """
    Visualize the analysis and synthesis windows and their overlap–add behavior.
    
    The following plots are generated:
      1) Comparison of a single-frame analysis and synthesis window.
      2) Sum of overlapped analysis windows (which may exceed 1.0).
      3) Sum of overlapped weighted windows (analysis * synthesis), which should be near 1.0
         if the WOLA design is correct.
    """
    L = len(analysis_window)
    hop = int(L * (1 - overlap))
    # Number of overlapping windows (e.g., 4 for 75% overlap)
    K = int(round(1.0 / (1.0 - overlap)))
    
    plt.figure(figsize=(10, 10))
    
    # 1) Analysis vs. Synthesis (Single Frame)
    plt.subplot(3, 1, 1)
    plt.title("Analysis vs. Synthesis Window (Single Frame)")
    plt.plot(analysis_window, label="Analysis (Blackman–Harris)", color="tab:blue")
    plt.plot(synthesis_window, label="Synthesis (WOLA)", color="tab:orange")
    plt.xlabel("Sample index")
    plt.ylabel("Amplitude")
    plt.legend(loc="best")
    
    # 2) Summation of Overlapped Analysis Windows Alone
    total_len = L + (K - 1) * hop
    analysis_sum = np.zeros(total_len, dtype=np.float32)
    for k in range(K):
        start_idx = k * hop
        analysis_sum[start_idx:start_idx + L] += analysis_window
    plt.subplot(3, 1, 2)
    plt.title(f"Sum of {K} Overlapped Analysis Windows at {overlap*100:.0f}% Overlap")
    plt.plot(analysis_sum, color="tab:blue", label="Analysis sum")
    plt.xlabel("Sample index")
    plt.ylabel("Amplitude")
    plt.legend(loc="best")
    
    # 3) Summation of Overlapped Weighted Windows (Analysis * Synthesis)
    weighted_sum = np.zeros(total_len, dtype=np.float32)
    w_combined = analysis_window * synthesis_window  # per-sample weighting
    for k in range(K):
        start_idx = k * hop
        weighted_sum[start_idx:start_idx + L] += w_combined
    plt.subplot(3, 1, 3)
    plt.title(f"Sum of {K} Overlapped Weighted Windows (Analysis*Synthesis)")
    plt.plot(weighted_sum, color="tab:orange", label="Weighted sum")
    plt.xlabel("Sample index")
    plt.ylabel("Amplitude")
    plt.legend(loc="best")
    
    plt.tight_layout()
    plt.show()

###############################################################################
# Updated Demo Main (Graphing Version)
###############################################################################
def main():
    """
    Demo routine that:
      • Loads an input audio file (in/eyes.wav).
      • Forces stereo (duplicating mono if needed).
      • Defines example frequency band edges.
      • Chains per-band extractors using the Blackman–Harris analysis window.
      • Optionally visualizes the analysis and synthesis windows.
      • Processes the bands in parallel to extract center, left, and right components.
      • Graphs the resulting upmixed signal vs. the original stereo sum in both
        the time and frequency domains.
    """
    # Define the input file path.
    in_dir = "in"
    infile = "eyes.wav"
    in_path = os.path.join(in_dir, infile)
    if not os.path.isfile(in_path):
        raise FileNotFoundError(f"Input file not found: {in_path}")
    
    # Load audio.
    wave, sr = sf.read(in_path)
    print(f"Loaded '{in_path}' with sample rate {sr} and shape {wave.shape}")
    
    # Ensure the audio is stereo (if mono, duplicate the channel).
    if wave.ndim == 1:
        wave = np.column_stack([wave, wave])
    L = wave[:, 0]
    R = wave[:, 1]
    
    # Define frequency band edges (in Hz) and STFT parameters.
    band_edges = [0.0, 40.0, 200.0, 2000.0]
    overlap = 0.75  # 75% overlap for STFT
    window_func = make_blackman_harris  # Primary analysis window
    
    # Build per-band extractors using the defined band edges.
    band_extractors = chain_bands(
        band_edges,
        overlap=overlap,
        window_func=window_func,
        sr=sr,
        xover_mode="raised_cosine",
    )
    
    # OPTIONAL: Visualize windows for the first band.
    if band_extractors:
        analysis_win = band_extractors[0].analysis_window
        synthesis_win = band_extractors[0].synthesis_window
        visualize_windows(analysis_win, synthesis_win, overlap)
    
    # Process the full-length signals in parallel over all bands.
    final_center, final_left, final_right = extract_center_left_right_multi_band_in_memory(
        L, R, sr, band_extractors
    )
    
    # Compare the upmixed sum (left side + center + right side) with the original stereo sum.
    upmix_sum = final_left + final_center + final_right
    orig_sum = L + R
    
    # Normalize both outputs for fair comparison.
    upmix_norm = upmix_sum / (np.max(np.abs(upmix_sum)) + 1e-12)
    orig_norm = orig_sum / (np.max(np.abs(orig_sum)) + 1e-12)
    
    time_axis = np.arange(len(upmix_norm)) / sr
    
    # --- Graphing ---
    plt.figure(figsize=(12, 8))
    
    # (1) Time–Domain Comparison
    plt.subplot(2, 1, 1)
    plt.plot(time_axis, upmix_norm, label="Upmix (L + C + R)", color="tab:blue")
    plt.plot(time_axis, orig_norm, label="Original (L + R)", color="tab:orange", alpha=0.75)
    plt.xlabel("Time (s)")
    plt.ylabel("Normalized Amplitude")
    plt.title("Time Domain Comparison")
    plt.legend(loc="upper right")
    
    # (2) Frequency–Domain Comparison (Magnitude Spectrum)
    n = len(upmix_norm)
    freq_axis = np.linspace(0, sr / 2, n // 2 + 1)
    upmix_fft = np.abs(np.fft.rfft(upmix_norm))
    orig_fft = np.abs(np.fft.rfft(orig_norm))
    
    plt.subplot(2, 1, 2)
    plt.semilogy(freq_axis, upmix_fft, label="Upmix Spectrum", color="tab:blue")
    plt.semilogy(freq_axis, orig_fft, label="Original Spectrum", color="tab:orange", alpha=0.75)
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Magnitude")
    plt.title("Frequency Domain Comparison")
    plt.legend(loc="upper right")
    
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
