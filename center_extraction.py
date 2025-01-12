#!/usr/bin/env python3
"""
Upmix: center_extraction.py

This module provides a multi-band STFT system for stereo audio. It:
1) Allows for user-defined high-pass crossover frequencies (band_edges), creating consecutive frequency bands up to sr/2.
2) Automatically computes a suitable STFT block size (power of two) for each band.
3) Applies raised-cosine crossovers or hard-zero filtering in the frequency domain.
4) Extracts left (Ls), center (C), and right (Rs) signals per band, then sums them across bands.
5) Uses ThreadPoolExecutor to process each band in parallel, leveraging multi-core CPUs.

Usage:
   - Import and call the functions to chain bands and run multi-band extraction, or run `main()` for a simple demo with input from `in/eyes.wav`.
   - The final output is a stereo file with upmixed vs. original sums, or you can build custom usage scenarios.
"""

import os
import math
import numpy as np
import soundfile as sf
from typing import Callable, List
from concurrent.futures import ThreadPoolExecutor  # For parallel band processing

###############################################################################
# Constants
###############################################################################
EPS = 1e-12

###############################################################################
# Window Functions
###############################################################################
def make_sqrt_hann(N: int) -> np.ndarray:
    """sqrt(Hann), common for 50% overlap."""
    h = np.hanning(N)
    return np.sqrt(h).astype(np.float32)

def make_hann(N: int) -> np.ndarray:
    """Plain Hann; often used if doing 75% overlap for analysis."""
    return np.hanning(N).astype(np.float32)

def make_blackman(N: int) -> np.ndarray:
    """Blackman window."""
    return np.blackman(N).astype(np.float32)

def make_hamming(N: int) -> np.ndarray:
    """Hamming window."""
    return np.hamming(N).astype(np.float32)

def make_blackman_harris(N: int) -> np.ndarray:
    """Blackman-Harris window."""
    n = np.arange(N)
    a0, a1, a2, a3 = 0.35875, 0.48829, 0.14128, 0.01168
    w = (
        a0
        - a1 * np.cos(2 * np.pi * n / (N - 1))
        + a2 * np.cos(4 * np.pi * n / (N - 1))
        - a3 * np.cos(6 * np.pi * n / (N - 1))
    )
    return w.astype(np.float32)

def make_rect(N: int) -> np.ndarray:
    """Rectangular (boxcar) window."""
    return np.ones(N, dtype=np.float32)

###############################################################################
# STFT Helpers
###############################################################################
def forward_stft(block: np.ndarray, window: np.ndarray) -> np.ndarray:
    wl = block * window
    return np.fft.rfft(wl)

def inverse_stft(spec: np.ndarray, window: np.ndarray) -> np.ndarray:
    rec = np.fft.irfft(spec).astype(np.float32)
    rec *= window
    return rec

###############################################################################
# freq_to_bin Utility
###############################################################################
def freq_to_bin(freq_hz: float, sr: float, fft_size: int) -> int:
    """
    Convert frequency in Hz to rFFT bin index (0-based).
    """
    bin_spacing = sr / float(fft_size)
    return int(round(freq_hz / bin_spacing))

###############################################################################
# Compute Block Size from Band's Lower Bound
###############################################################################
def next_power_of_2(x: int) -> int:
    """Return the smallest power of 2 >= x."""
    if x < 1:
        return 1
    power = 1
    while power < x:
        power <<= 1
    return power

def compute_block_size_for_low_freq(f_low: float, sr: float, max_block_size: int = 2**16) -> int:
    """
    Pick a block size so that: (sr / block_size) * 20 < f_low.
    Round up to the next power of 2. If f_low <= 0, or needed size > max_block_size,
    clamp to max_block_size.
    """
    if f_low <= 0.0:
        return max_block_size

    threshold = (sr * 20) / f_low
    candidate_blk = next_power_of_2(int(np.ceil(threshold)))

    # Clamp if needed
    if candidate_blk > max_block_size:
        candidate_blk = max_block_size

    return candidate_blk

###############################################################################
# Crossover width
###############################################################################
def hp_freq_to_crossover_width(hp_freq: float) -> float:
    """
    xover_width = hp_freq/4 by default, e.g., 2000 Hz => 500 Hz fade zone.
    """
    return hp_freq * 0.25

###############################################################################
# MULTI-BAND EXTRACTOR WITH CONDITIONAL RAISED-COSINE
###############################################################################
class MultiBandExtractorAccu:
    """
    Per-band STFT-based extractor that:
    - Limits/fades frequencies outside [f_low, f_high].
    - Extracts center via cross-spectral analysis, with left and right side signals.
    - Overlap-adds time blocks to maintain continuous output.

    xover_mode can be:
      - "hard_zero" => zero out bins outside [f_low, f_high]
      - "raised_cosine" => half-cosine fade near f_low/f_high
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
            raise ValueError("Overlap too large; hop < 1 is not allowed.")

        self.window = window_func(block_size)
        self.sr     = sr
        self.f_low  = f_low
        self.f_high = f_high

        self.xover_mode          = xover_mode
        self.xover_width_low_hz  = xover_width_low_hz
        self.xover_width_high_hz = xover_width_high_hz

        # Overlap-add accumulators
        self.accumC = np.zeros(block_size, dtype=np.float32)
        self.accumL = np.zeros(block_size, dtype=np.float32)
        self.accumR = np.zeros(block_size, dtype=np.float32)

    def _hard_zero_filter(self, specL, specR, bin_low, bin_high):
        specL[:bin_low] = 0
        specR[:bin_low] = 0
        specL[bin_high+1:] = 0
        specR[bin_high+1:] = 0

    def _raised_cosine_filter(self, specL, specR, bin_low, bin_high, fft_size):
        """
        Half-cosine fade near f_low or f_high, skipping fade if f_low <= 0 or f_high >= sr/2.
        """
        n_bins = len(specL)
        if bin_low > bin_high:
            bin_low, bin_high = bin_high, bin_low
        bin_low  = max(bin_low, 0)
        bin_high = min(bin_high, n_bins - 1)

        # If passband is empty => zero everything
        if bin_low > bin_high:
            specL[:] = 0
            specR[:] = 0
            return

        # Convert fade widths (Hz->bins)
        fade_bins_low  = freq_to_bin(self.xover_width_low_hz,  self.sr, fft_size)
        fade_bins_high = freq_to_bin(self.xover_width_high_hz, self.sr, fft_size)

        # Fade in from bottom (unless f_low <= 0)
        if self.f_low > 0:
            fade_in_start = max(0, bin_low - fade_bins_low)
            specL[:fade_in_start] = 0
            specR[:fade_in_start] = 0

            if fade_in_start < bin_low:
                fade_in_len = bin_low - fade_in_start
                for i in range(fade_in_len):
                    idx = fade_in_start + i
                    x = (i + 0.5) / fade_in_len
                    alpha = 0.5 * (1.0 - np.cos(np.pi * x))  # 0..1
                    specL[idx] *= alpha
                    specR[idx] *= alpha

        # Fade out near top (unless f_high >= sr/2)
        if self.f_high < self.sr * 0.5:
            fade_out_start = bin_high + 1
            fade_out_end   = fade_out_start + fade_bins_high
            if fade_out_start < n_bins:
                fade_out_end = min(fade_out_end, n_bins)
                fade_out_len = fade_out_end - fade_out_start
                for i in range(fade_out_len):
                    idx = fade_out_start + i
                    x = (i + 0.5) / fade_out_len
                    alpha = 0.5 * (1.0 + np.cos(np.pi * x))  # 1..0
                    specL[idx] *= alpha
                    specR[idx] *= alpha

                if fade_out_end < n_bins:
                    specL[fade_out_end:] = 0
                    specR[fade_out_end:] = 0

    def _band_limit(self, specL, specR):
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
            self._hard_zero_filter(specL, specR, bin_low, bin_high)

    def process_stereo_chunk(self, blkL, blkR):
        # Forward STFT
        specL = forward_stft(blkL, self.window)
        specR = forward_stft(blkR, self.window)

        # Limit or fade freq bins
        self._band_limit(specL, specR)

        # Cross-spectral L/C/R
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

        # Inverse STFT
        rec_center = inverse_stft(spec_center, self.window)
        rec_left   = inverse_stft(spec_left,   self.window)
        rec_right  = inverse_stft(spec_right,  self.window)

        # Overlap-add accumulators
        self.accumC += rec_center
        self.accumL += rec_left
        self.accumR += rec_right

        # Output chunk = first hop_size samples
        out_c = self.accumC[:self.hop_size].copy()
        out_l = self.accumL[:self.hop_size].copy()
        out_r = self.accumR[:self.hop_size].copy()

        # Shift accumulators
        self.accumC[:-self.hop_size] = self.accumC[self.hop_size:]
        self.accumL[:-self.hop_size] = self.accumL[self.hop_size:]
        self.accumR[:-self.hop_size] = self.accumR[self.hop_size:]

        # Clear the last region
        self.accumC[-self.hop_size:] = 0
        self.accumL[-self.hop_size:] = 0
        self.accumR[-self.hop_size:] = 0

        return out_c, out_l, out_r

    def flush_final(self):
        leftover_c = self.accumC.copy()
        leftover_l = self.accumL.copy()
        leftover_r = self.accumR.copy()

        self.accumC[:] = 0
        self.accumL[:] = 0
        self.accumR[:] = 0

        return leftover_c, leftover_l, leftover_r

    def process_all_blocks(self, L: np.ndarray, R: np.ndarray):
        """
        Process the entire input (L,R) for this band, returning band-limited
        center/left/right signals (time domain).
        """
        N = len(L)
        leftover_len = self.block_size - self.hop_size
        needed       = leftover_len
        num_hops     = math.ceil((N - needed) / self.hop_size)
        padded_len   = num_hops * self.hop_size + needed
        pad_amt      = max(0, padded_len - N)

        L_pad = np.pad(L, (0, pad_amt), mode='constant')
        R_pad = np.pad(R, (0, pad_amt), mode='constant')

        out_c = []
        out_l = []
        out_r = []
        idx   = 0

        while idx < len(L_pad):
            blkL = L_pad[idx : idx + self.block_size]
            blkR = R_pad[idx : idx + self.block_size]
            if len(blkL) < self.block_size:
                blkL = np.pad(blkL, (0, self.block_size - len(blkL)), mode='constant')
                blkR = np.pad(blkR, (0, self.block_size - len(blkR)), mode='constant')

            c_chunk, l_chunk, r_chunk = self.process_stereo_chunk(blkL, blkR)
            out_c.append(c_chunk)
            out_l.append(l_chunk)
            out_r.append(r_chunk)

            idx += self.hop_size

        # Final flush
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
    Multi-band approach with thread-based parallelization:
    - Each band_extractor processes the full input.
    - We sum L/C/R from each band to form the final signals.
    """
    import concurrent.futures

    futures = []
    with concurrent.futures.ThreadPoolExecutor() as executor:
        for bex in band_extractors:
            fut = executor.submit(bex.process_all_blocks, L, R)
            futures.append(fut)

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
# chain_bands
###############################################################################
def chain_bands(
    band_edges: List[float],
    overlap: float,
    window_func: Callable[[int], np.ndarray],
    sr: float,
    xover_mode: str = "raised_cosine",
):
    """
    Break the spectrum into consecutive [f_low..f_high] bands,
    ensuring the top band extends to sr/2 if not explicitly covered.
    Each band is assigned a block_size based on f_low.

    Example:
      band_edges = [0, 40, 200, 2000] => Bands: [0..40], [40..200], [200..2000], [2000..sr/2].
    """
    if band_edges[-1] < (sr / 2.0):
        band_edges = list(band_edges) + [sr / 2.0]

    extractors = []
    prev_xover_high = 0.0

    for i in range(len(band_edges) - 1):
        f_low  = band_edges[i]
        f_high = band_edges[i + 1]

        block_size = compute_block_size_for_low_freq(f_low, sr)
        xover_low  = prev_xover_high
        xover_high = hp_freq_to_crossover_width(f_high)

        print(
            f"[Band {i+1}] f_low={f_low:.1f} Hz, "
            f"f_high={f_high:.1f} Hz, block_size={block_size}, "
            f"xover_low={xover_low:.1f}, xover_high={xover_high:.1f}"
        )

        ext = MultiBandExtractorAccu(
            block_size        = block_size,
            overlap           = overlap,
            window_func       = window_func,
            f_low             = f_low,
            f_high            = f_high,
            sr                = sr,
            xover_mode        = xover_mode,
            xover_width_low_hz  = xover_low,
            xover_width_high_hz = xover_high
        )
        extractors.append(ext)

        prev_xover_high = xover_high

    return extractors

###############################################################################
# Main
###############################################################################
def main():
    in_dir = "in"
    out_dir = "out"
    os.makedirs(out_dir, exist_ok=True)

    infile = "eyes.wav"
    in_path = os.path.join(in_dir, infile)
    if not os.path.isfile(in_path):
        raise FileNotFoundError(f"File not found: {in_path}")

    wave, sr = sf.read(in_path)
    print(f"Loaded '{in_path}', sr={sr}, shape={wave.shape}")

    if wave.ndim == 1:
        wave = np.column_stack([wave, wave])
    L = wave[:, 0]
    R = wave[:, 1]

    band_edges = [0.0, 40.0, 200.0, 2000.0]
    overlap    = 0.75
    window_func = make_blackman_harris

    band_extractors = chain_bands(
        band_edges,
        overlap=overlap,
        window_func=window_func,
        sr=sr,
        xover_mode="raised_cosine",
    )

    final_center, final_left, final_right = extract_center_left_right_multi_band_in_memory(
        L, R, sr, band_extractors
    )

    upmix_sum = final_left + final_center + final_right
    orig_sum  = L + R
    N = min(len(upmix_sum), len(orig_sum))
    upmix_slice = upmix_sum[:N].copy()
    orig_slice  = orig_sum[:N].copy()

    peak_upmix = np.max(np.abs(upmix_slice))
    if peak_upmix > 0:
        upmix_slice /= peak_upmix

    peak_orig = np.max(np.abs(orig_slice))
    if peak_orig > 0:
        orig_slice /= peak_orig

    stereo_out = np.column_stack((upmix_slice, orig_slice))

    band_desc_list = [
        f"b{bex.block_size}({int(bex.f_low)}-{int(bex.f_high)})"
        for bex in band_extractors
    ]
    band_info_str = "_".join(band_desc_list)

    out_fname = f"{os.path.splitext(infile)[0]}_parallel_{band_info_str}_ov{overlap:.2f}_bmh.wav"
    out_path  = os.path.join(out_dir, out_fname)

    sf.write(out_path, stereo_out, sr)
    print(
        f"\nWrote multi-band L/C/R vs Original sum =>\n'{out_path}'\n"
        "Left channel  = Upmix (Lside + Center + Rside) [Normalized]\n"
        "Right channel = Original sum (L + R) [Normalized]"
    )

if __name__ == "__main__":
    main()
