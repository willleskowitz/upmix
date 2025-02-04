# Upmix

**Upmix** is an open‐source project dedicated to prototyping algorithms for stereo localization and transforming conventional two‐channel audio into a more immersive, multi‐channel experience.

## Multi-Band Left, Center, and Right Extraction

Upmix provides a collection of tools for STFT-based multi-band audio processing. The project includes two implementations:

- **Python Prototype:**  
  A set of scripts for offline STFT-based multi-band processing that extract left-side (Ls), center (C), and right-side (Rs) signals from stereo audio. It offers flexible crossover definitions, parallel processing, and multiple export modes.

- **C++ Real-Time Program (Bela):**  
  A real-time implementation optimized for [Bela](https://github.com/BelaPlatform/Bela) (BeagleBone Black). This version uses dynamic frequency resolution, raised cosine smoothing, and efficient memory management to process stereo input in real time and generate upmixed outputs.

## Key Features

- **Dynamic Frequency Resolution:**  
  Computes per-band STFT sizes using the same algorithm as the Python prototype. A configurable threshold multiplier (default set to 32) is used so that lower frequencies receive higher resolution (up to `hwBlockSize * 4`).

- **Raised Cosine Smoothing:**  
  A raised cosine filter is applied near the crossover frequencies to emulate the smooth summing of a fourth-order Linkwitz–Riley (LR) crossover. The fraction used for smoothing is controlled by a global constant (`XO_FRACTION`, default 0.25) that can be adjusted as needed.

- **Optimized Memory Management:**  
  Circular buffers are allocated dynamically based on the actual STFT size and the required number of passes (plus a safety margin), minimizing wasted memory while ensuring all necessary data is available. Other temporary arrays are sized relative to `MAX_STFT_SIZE`, the maximum allowed.

- **Real-Time Efficiency:**  
  The C++ code is optimized for the single-core BeagleBone Black used by Bela. All processing is performed in C++ with preallocated buffers to avoid runtime memory allocation, keeping latency low.

## How It Works

- **Input:**  
  The system reads stereo audio in real time from Bela’s audio inputs (or processes a stereo WAV file in the Python prototype).

- **Processing:**  
  The audio is split into multiple frequency bands defined by adjacent crossover frequencies. For each band, an STFT is computed, then processed in the frequency domain—including raised cosine smoothing and center extraction—before being inverted and recombined via overlap-add.

- **Output:**  
  The processed signals are routed to Bela’s outputs. In the current configuration, the left channel is computed as:
  
  ```cpp
  float valL = chunkL[i] + 0.5f * chunkC[i];
  ```
  and the right channel as:
  ```cpp
  float valR = chunkR[i] + 0.5f * chunkC[i];
  ```
  This means that the left output consists of the left-side signal plus half the center signal, and the right output is the right-side signal plus half the center signal.  

  **To route the center (C) signal to a separate output channel (i.e., create a true three-channel system), modify these formulas accordingly in the source code.**
  
  ## Installation and Build
  
  ### Python Prototype
  1. **Clone** the repository.
  2. **Install dependencies.** Note `scipy` is only required for FIR filter design if you use `filter_design.py`.
     
     ```bash
     pip install numpy soundfile scipy
     ```
  3. **Create** `in/` and `out/` directories in the repository for storing input and output files.

  ### C++ Real-Time Program (Bela)
  
  1. **Copy** the C++ files (e.g., `upmix.cpp`) into a new project in the Bela IDE.
  2. **Configure** the project settings as needed. A block size of 2048 and a sample rate of 48 kHz are recommended.
  3. **Compile and upload** the code to your Bela board. By default, channels 0 and 1 are used for input and output.

  ## Usage
  
  ### Python Prototype
  
  - **Place** your stereo WAV file in the `in/` folder.
  - **Edit** `main.py` to set your input file, export mode, and crossover frequencies.
  - **Run:**
    
    ```bash
    python main.py
    ```
    Check the `out/` folder for the resulting WAV files.

  ### C++ Real-Time Program (Bela)
  
  - **Adjust** the crossover frequencies and threshold multiplier (if necessary) in the source code.
  - **Build and upload** the project using the Bela IDE.
  - The program processes stereo input in real time and outputs the upmixed channels as described above.
  
  ## Notes and Tips
  
  ### Parallel Processing (Python)
  - The Python scripts utilize `ThreadPoolExecutor` for parallel band processing, which can accelerate processing on multi-core machines.
  
  ### Windowing and Normalization
  - Both implementations are designed to preserve the overall signal balance relative to the input.
  - The **Python prototype** employs a fully flexible WOLA framework—with adjustable overlap and window parameters—to achieve near-perfect reconstruction and maintain proper normalization.
  - In contrast, the **C++ implementation** uses a fixed 75% overlap WOLA approach using Blackman–Harris windows. If further normalization adjustments are required, you can tweak the synthesis window parameters or add additional scaling factors.
  
  ### Real-Time Considerations
  - The C++ code is optimized for a single-core system. Monitor CPU usage and adjust parameters (such as STFT size or number of bands) if required.
  
  ## License
  
  Licensed under the [Apache License 2.0](LICENSE). See the LICENSE file for details.
