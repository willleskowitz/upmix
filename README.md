# Upmix

**Upmix** is an open-source project focused on prototyping algorithms for stereo localization and transforming conventional two-channel audio into a more immersive, multi-channel experience.

## Multi-Band Center Extraction and Upmix

A collection of Python scripts for **STFT-based multi-band audio processing**, particularly aimed at extracting left-side (Ls), center (C), and right-side (Rs) signals from stereo audio. Includes options for parallel (multi-threaded) band processing, flexible crossover definitions, and multiple export modes.

---

## Contents

- **`main.py`**  
  - Loads a stereo WAV from the `in/` folder.  
  - Defines crossover points (`band_edges`) and extracts Ls, C, Rs.  
  - Depending on `export_mode` (`"AB"`, `"split"`, or `"stereo_sum"`), saves different output WAVs:
    - **AB**: One 2-channel file (Left = Ls + C + Rs, Right = L + R).
    - **split**: Three separate stereo files isolating Ls, C, and Rs with panning.
    - **stereo_sum**: One stereo file with (Left = Ls + C, Right = C + Rs).

- **`center_extraction.py`**  
  - Core multi-band STFT logic:
    - **`chain_bands`**: Creates per-band STFT extractors based on user-specified band edges.  
    - **`MultiBandExtractorAccu`**: Handles STFT, frequency-domain band limiting/fading, cross-spectral center extraction, and iSTFT with overlap-add.  
    - **`extract_center_left_right_multi_band_in_memory`**: Runs each band in parallel (via `ThreadPoolExecutor`) and sums partial signals for the final wideband Ls/C/R.

- **`filter_design.py`**  
  - Contains example FIR filter design (approx. 4th-order Linkwitz-Riley) and a helper to apply FIR taps.
  - Not mandatory for STFT-based extraction, but useful if you explore FIR-based crossovers.

---

## Installation

1. **Clone** this repository or download the source.  
2. **Install dependencies**:
   ```bash
   pip install numpy soundfile scipy
**Note**: `scipy` is only required if you plan to use `filter_design.py` for FIR creation.

## Usage

1. **Place** your stereo WAV file in the `in/` folder.  
2. **Open** `main.py`:
   - Change `in_filename` to your WAV’s name.
   - Adjust `export_mode` to `"AB"`, `"split"`, or `"stereo_sum"`.
   - Optionally modify `band_edges` to set different crossover frequencies.
3. **Run**:
   ```bash
   python main.py
Check the `out/` folder for your resulting WAV(s).

## Notes and Tips

### Parallel Processing
`center_extraction.py` uses `ThreadPoolExecutor` for per-band parallelism, potentially speeding up processing on multi-core CPUs.

### Normalization
By default, the scripts ensure that Ls, C, Rs do not exceed the original input peak amplitude, preserving relative levels. Adjust if you prefer a different normalization scheme.

### Real-Time Usage
While this project currently operates offline, the **long-term goal** is to support real-time workflows.

## License

Licensed under the [Apache License 2.0](LICENSE). See the `LICENSE` file for details.
