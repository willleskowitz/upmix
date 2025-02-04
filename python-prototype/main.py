#!/usr/bin/env python3
"""
Upmix: main.py

This script:
1) Loads a stereo WAV from the 'in/' directory (any filename).
2) Defines crossover frequencies and performs STFT-based multi-band extraction to obtain Ls, C, and Rs.
3) Depending on export_mode, writes one or more WAV files:
   - "AB": 2-channel file (Left = Ls + C + Rs, Right = L + R).
   - "split": three separate stereo files (Ls/C/Rs each isolated).
   - "stereo_sum": single stereo file (Left = Ls + C/2, Right = Rs + C/2).
4) Scales Ls, C, and Rs so they do not exceed the original peak amplitude.

Usage:
   - Edit 'in_filename' and 'export_mode' as desired.
   - Run: python main.py
   - Check the 'out/' folder for results.
"""

import os
import numpy as np
import soundfile as sf
import center_extraction as ce  # local import

def main():
    # --------------------------------------------------------------------------
    # User-Config: Adjust these as needed.
    # --------------------------------------------------------------------------
    in_filename = "eyes.wav"  # change to your WAV name in 'in/' folder
    export_mode = "stereo_sum"  # "AB", "split", or "stereo_sum"

    in_dir = "in"
    out_dir = "out"
    os.makedirs(out_dir, exist_ok=True)

    # --------------------------------------------------------------------------
    # 1) Load input
    # --------------------------------------------------------------------------
    in_path = os.path.join(in_dir, in_filename)
    if not os.path.isfile(in_path):
        raise FileNotFoundError(f"File not found: {in_path}")

    wave, sr = sf.read(in_path)
    print(f"Loaded '{in_path}', sr={sr}, shape={wave.shape}")

    # Ensure stereo
    if wave.ndim == 1:
        wave = np.column_stack([wave, wave])
    L = wave[:, 0]
    R = wave[:, 1]

    # Measure peak amplitude of the original wave
    peak_in = np.max(np.abs(wave))  # across both channels
    if peak_in <= 0.0:
        peak_in = 1e-9  # avoid divide-by-zero if the file is silent

    # --------------------------------------------------------------------------
    # 2) Define crossovers and build the multi-band extractors
    # --------------------------------------------------------------------------
    # Example: a handful of crossovers for demonstration
    # band_edges = [0, 40, 200, 2000]
    band_edges = [0, 30, 120, 480, 1920, 7680]

    overlap     = 0.75
    window_func = ce.make_blackman_harris  # or ce.make_hann, etc.

    band_extractors = ce.chain_bands(
        band_edges,
        overlap=overlap,
        window_func=window_func,
        sr=sr,
        xover_mode="raised_cosine",
    )

    # --------------------------------------------------------------------------
    # 3) Extract Ls, C, Rs
    # --------------------------------------------------------------------------
    final_center, final_left, final_right = ce.extract_center_left_right_multi_band_in_memory(
        L, R, sr, band_extractors
    )

    # --------------------------------------------------------------------------
    # 4) Compute a single scale factor so Ls, C, Rs won't exceed the original peak
    # --------------------------------------------------------------------------
    raw_peak_l = np.max(np.abs(final_left))
    raw_peak_c = np.max(np.abs(final_center))
    raw_peak_r = np.max(np.abs(final_right))
    overall_peak = max(raw_peak_l, raw_peak_c, raw_peak_r, 1e-9)

    scale_factor = peak_in / overall_peak
    print(f"Original peak = {peak_in:.4f}, L/C/R peak = {overall_peak:.4f}")
    print(f"Applying scale_factor = {scale_factor:.4f}")

    # Apply scaling
    final_left   *= scale_factor
    final_center *= scale_factor
    final_right  *= scale_factor

    # --------------------------------------------------------------------------
    # 5) Build and write the output(s) depending on export_mode
    # --------------------------------------------------------------------------
    band_desc_list = [
        f"b{bex.block_size}({int(bex.f_low)}-{int(bex.f_high)})"
        for bex in band_extractors
    ]
    band_info_str = "_".join(band_desc_list)

    base_in_name = os.path.splitext(in_filename)[0]

    if export_mode == "AB":
        # 2-ch A/B
        upmix_sum = final_left + final_center + final_right
        orig_sum  = L + R
        N = min(len(upmix_sum), len(orig_sum))
        ab_stereo = np.column_stack([upmix_sum[:N], orig_sum[:N]])

        out_fname = f"{base_in_name}_AB_{band_info_str}_ov{overlap:.2f}.wav"
        out_path  = os.path.join(out_dir, out_fname)
        sf.write(out_path, ab_stereo, sr)

        print(f"[AB] Wrote 2-ch => {out_path}\n"
              "  Left  = (Ls + C + Rs)\n"
              "  Right = (L + R)\n")

    elif export_mode == "split":
        # Three separate stereo files
        Ls_stereo = np.column_stack([final_left, np.zeros_like(final_left)])
        C_stereo  = np.column_stack([final_center, final_center])
        Rs_stereo = np.column_stack([np.zeros_like(final_right), final_right])

        Ls_path = os.path.join(out_dir, f"{base_in_name}_Ls_{band_info_str}.wav")
        sf.write(Ls_path, Ls_stereo, sr)
        print(f"[Split] Wrote => {Ls_path} (Left=Ls, Right=0)")

        C_path = os.path.join(out_dir, f"{base_in_name}_C_{band_info_str}.wav")
        sf.write(C_path, C_stereo, sr)
        print(f"[Split] Wrote => {C_path} (Left=C, Right=C)")

        Rs_path = os.path.join(out_dir, f"{base_in_name}_Rs_{band_info_str}.wav")
        sf.write(Rs_path, Rs_stereo, sr)
        print(f"[Split] Wrote => {Rs_path} (Left=0, Right=Rs)")

    elif export_mode == "stereo_sum":
        # Single stereo file with (Left = Ls + C/2, Right = Rs + C/2)
        left_ch  = final_left + 0.5*final_center
        right_ch = final_right+ 0.5*final_center 

        N = min(len(left_ch), len(right_ch))
        stereo_sum = np.column_stack([left_ch[:N], right_ch[:N]])

        out_fname = f"{base_in_name}_Sum_{band_info_str}_ov{overlap:.2f}.wav"
        out_path  = os.path.join(out_dir, out_fname)
        sf.write(out_path, stereo_sum, sr)

        print(f"[StereoSum] Wrote 2-ch => {out_path}\n"
              "  Left  = (Ls + C/2)\n"
              "  Right = (Rs + C/2)\n")

    else:
        print(f"Unknown export_mode '{export_mode}' -- no files written.")

    print("Done.")

if __name__ == "__main__":
    main()
