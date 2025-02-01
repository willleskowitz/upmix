// MultiBandUpmix.cpp
// A multi-band upmixing algorithm for Bela with dynamic frequency resolution
// and robust overlapping STFT processing. The code computes per-band STFT sizes
// based on the desired frequency resolution (using Python-like logic) and
// processes each band independently.
// Author: [Your Name]
// Date: [Today's Date]

#include <Bela.h>
#include <libraries/Fft/Fft.h>
#include <cmath>
#include <stdexcept>
#include <algorithm>
#include <cstdio>

//======================================
// Configurable compile-time limits
//======================================
static constexpr unsigned int MAX_STFT_SIZE    = 8192;   // Maximum STFT block size (adjust as needed)
static constexpr unsigned int RING_BUFFER_SIZE = 2 * MAX_STFT_SIZE; 
static constexpr unsigned int MAX_BUFFER_SIZE  = MAX_STFT_SIZE;
static constexpr float EPS = 1e-12f;

//-------------------------------------
// Helper: next power of 2
//-------------------------------------
// Returns the smallest power of 2 greater than or equal to x.
static unsigned int nextPowerOf2(unsigned int x)
{
    unsigned int power = 1;
    while(power < x)
        power <<= 1;
    return power;
}

//-------------------------------------
// Helper: map frequency (Hz) to FFT bin
//-------------------------------------
static unsigned int freqToBin(float freqHz, float sr, unsigned int fftSize)
{
    float binF = freqHz * fftSize / sr;
    if(binF < 0.f)
        binF = 0.f;
    float maxBin = (float)(fftSize / 2);
    if(binF > maxBin)
        binF = maxBin;
    return (unsigned int)std::lround(binF);
}

//-------------------------------------
// Hard-zero frequency outside [fLow, fHigh]
//-------------------------------------
// Zeros out the frequency bins (for both left and right channels) outside the desired range.
static void bandLimitFreqDomain(float fLow, float fHigh,
                                float* reL, float* imL,
                                float* reR, float* imR,
                                float sr, unsigned int stftSize)
{
    unsigned int nBins = stftSize / 2 + 1;
    unsigned int binLow = freqToBin(fLow, sr, stftSize);
    unsigned int binHigh = freqToBin(fHigh, sr, stftSize);
    if(binLow > binHigh)
        std::swap(binLow, binHigh);
    for(unsigned int k = 0; k < nBins; k++){
        if(k < binLow || k > binHigh){
            reL[k] = 0.f; imL[k] = 0.f;
            reR[k] = 0.f; imR[k] = 0.f;
        }
    }
}

//-------------------------------------
// Blackman-Harris Window Generator
//-------------------------------------
// Generates a Blackman-Harris window of a given size.
static void makeBlackmanHarris(float* w, unsigned int size)
{
    constexpr float a0 = 0.35875f;
    constexpr float a1 = 0.48829f;
    constexpr float a2 = 0.14128f;
    constexpr float a3 = 0.01168f;
    for(unsigned int n = 0; n < size; n++){
        float ratio = (float)n / (float)(size - 1);
        w[n] = a0
             - a1 * cosf(2.f * (float)M_PI * ratio)
             + a2 * cosf(4.f * (float)M_PI * ratio)
             - a3 * cosf(6.f * (float)M_PI * ratio);
    }
}

//--------------------------------------------------------------------------------
// CircularBuffer Class
//--------------------------------------------------------------------------------
// A fixed-size circular buffer that maintains a persistent read pointer so that
// each call to readBlock() advances by hopSize.
class CircularBuffer {
public:
    // Set up the circular buffer with the given size.
    void setup(unsigned int size)
    {
        if(size > RING_BUFFER_SIZE)
            throw std::runtime_error("Ring size > RING_BUFFER_SIZE");
        size_ = size;
        writePos_ = 0;
        readPos_ = 0; // separate read pointer
        fillCount_ = 0;
        for(unsigned int i = 0; i < RING_BUFFER_SIZE; i++){
            buffer_[i] = 0.f;
        }
    }
    // Write samples into the buffer.
    void writeSamples(const float* in, unsigned int numSamples)
    {
        for(unsigned int i = 0; i < numSamples; i++){
            buffer_[writePos_] = in[i];
            writePos_ = (writePos_ + 1) % size_;
        }
        fillCount_ += numSamples;
    }
    // Check if at least 'required' samples are available.
    bool canProcess(unsigned int required) const
    {
        return (fillCount_ >= required);
    }
    // Read a block of stftSize samples starting from the persistent read pointer,
    // then advance the read pointer by hopSize.
    void readBlock(float* out, unsigned int stftSize, unsigned int hopSize)
    {
        if(stftSize > size_)
            throw std::runtime_error("stftSize > circular buffer size");
        unsigned int currentRead = readPos_;
        for(unsigned int n = 0; n < stftSize; n++){
            out[n] = buffer_[(currentRead + n) % size_];
        }
        readPos_ = (readPos_ + hopSize) % size_;
        if(fillCount_ >= hopSize)
            fillCount_ -= hopSize;
        else
            fillCount_ = 0;
    }
private:
    float buffer_[RING_BUFFER_SIZE] = {0.f};
    unsigned int size_ = 0;
    unsigned int writePos_ = 0;
    unsigned int readPos_ = 0; // persistent read pointer
    unsigned int fillCount_ = 0;
};

//--------------------------------------------------------------------------------
// OverlapAdd Class
//--------------------------------------------------------------------------------
// Implements the accumulation and pop logic for WOLA reconstruction.
class OverlapAdd {
public:
    // Initialize the accumulator array to zero.
    void setup(unsigned int size)
    {
        if(size > MAX_STFT_SIZE)
            throw std::runtime_error("OverlapAdd size > MAX_STFT_SIZE");
        size_ = size;
        for(unsigned int i = 0; i < MAX_STFT_SIZE; i++){
            accum_[i] = 0.f;
        }
    }
    // Accumulate a new block (multiplied by the synthesis window) into the accumulator.
    void accumulate(const float* block, const float* window)
    {
        for(unsigned int n = 0; n < size_; n++){
            accum_[n] += block[n] * window[n];
        }
    }
    // Pop the first hopSize samples (which are now fully overlapped) and shift the accumulator.
    void popHop(unsigned int hopSize, float* out)
    {
        for(unsigned int i = 0; i < hopSize; i++){
            out[i] = accum_[i];
        }
        for(unsigned int i = hopSize; i < size_; i++){
            accum_[i - hopSize] = accum_[i];
        }
        for(unsigned int i = size_ - hopSize; i < size_; i++){
            accum_[i] = 0.f;
        }
    }
private:
    float accum_[MAX_STFT_SIZE] = {0.f};
    unsigned int size_ = 0;
};

//--------------------------------------------------------------------------------
// Overlap75UpmixBand Class
//--------------------------------------------------------------------------------
// This class implements one frequency band’s upmix processing using 75% overlap.
// It performs the following steps:
// 1. Reads stftSize samples from its circular buffer (advancing by hopSize each pass).
// 2. Applies the analysis window and computes the FFT.
// 3. Zeroes frequency bins outside [fLow, fHigh].
// 4. Performs the upmix algorithm (computing left-side, right-side, and center).
// 5. Computes the inverse FFT (with the synthesis window) and uses overlap-add to reconstruct time-domain output.
class Overlap75UpmixBand {
public:
    // Setup the band with the given hardware block size, sample rate, STFT size, and frequency range.
    void setup(unsigned int hwBlockSize, float sr,
               unsigned int stftSize, float fLow, float fHigh)
    {
        hwBlockSize_ = hwBlockSize;
        sr_ = sr;
        fLow_ = fLow;
        fHigh_ = fHigh;
        overlap_ = 0.75f;
        stftSize_ = stftSize;
        hopSize_ = (unsigned int)std::lround(stftSize_ * (1.f - overlap_));
        if(stftSize_ > MAX_STFT_SIZE)
            throw std::runtime_error("stftSize too large");

        // Setup FFT objects (for forward and inverse transforms)
        if(fwdL_.setup(stftSize_) != 0 || invL_.setup(stftSize_) != 0)
            throw std::runtime_error("FFT(L) failed");
        if(fwdR_.setup(stftSize_) != 0 || invR_.setup(stftSize_) != 0)
            throw std::runtime_error("FFT(R) failed");

        // Generate analysis and synthesis windows.
        makeBlackmanHarris(analysisWin_, stftSize_);
        makeBlackmanHarris(synthesisWin_, stftSize_);

        // Initialize the overlap-add accumulators.
        overlapAddL_.setup(stftSize_);
        overlapAddR_.setup(stftSize_);
        overlapAddC_.setup(stftSize_);

        // Setup circular buffers (sized generously to allow multiple passes).
        circBufL_.setup(stftSize_ * 8);
        circBufR_.setup(stftSize_ * 8);

        // Zero initialize frequency-domain and time-domain temporary arrays.
        for(unsigned int i = 0; i < (MAX_STFT_SIZE / 2 + 1); i++){
            reL_[i] = imL_[i] = 0.f;
            reR_[i] = imR_[i] = 0.f;
            reC_[i] = imC_[i] = 0.f;
            reLs_[i] = imLs_[i] = 0.f;
            reRs_[i] = imRs_[i] = 0.f;
        }
        for(unsigned int i = 0; i < MAX_STFT_SIZE; i++){
            timeDomainL_[i] = timeDomainR_[i] = timeDomainC_[i] = 0.f;
        }
    }
    // Feed new input samples into the circular buffers.
    void feed(const float* inL, const float* inR, unsigned int frames)
    {
        circBufL_.writeSamples(inL, frames);
        circBufR_.writeSamples(inR, frames);
    }
    // Returns true if enough samples have accumulated to process one full hardware block.
    bool canProcessHwBlock() const
    {
        unsigned int neededPasses = hwBlockSize_ / hopSize_;
        unsigned int neededSamples = stftSize_ * neededPasses;
        return (circBufL_.canProcess(neededSamples) &&
                circBufR_.canProcess(neededSamples));
    }
    // Process enough STFT passes to produce exactly one hardware block of output.
    void doOneHwBlock(float* outL, float* outR)
    {
        // Clear output buffer.
        for(unsigned int i = 0; i < hwBlockSize_; i++){
            outL[i] = 0.f;
            outR[i] = 0.f;
        }
        unsigned int numPasses = hwBlockSize_ / hopSize_;
        unsigned int writePos = 0;
        // For each pass, read input, process the STFT and inverse STFT,
        // then use overlap-add to extract hopSize samples.
        for(unsigned int pass = 0; pass < numPasses; pass++){
            circBufL_.readBlock(timeDomainL_, stftSize_, hopSize_);
            circBufR_.readBlock(timeDomainR_, stftSize_, hopSize_);
            // Apply analysis window and perform forward FFT.
            for(unsigned int n = 0; n < stftSize_; n++){
                fwdL_.td(n) = timeDomainL_[n] * analysisWin_[n];
                fwdR_.td(n) = timeDomainR_[n] * analysisWin_[n];
            }
            fwdL_.fft();
            fwdR_.fft();
            unsigned int nBins = stftSize_ / 2 + 1;
            for(unsigned int k = 0; k < nBins; k++){
                reL_[k] = fwdL_.fdr(k);  imL_[k] = fwdL_.fdi(k);
                reR_[k] = fwdR_.fdr(k);  imR_[k] = fwdR_.fdi(k);
            }
            // Zero out frequency bins outside the desired frequency range.
            bandLimitFreqDomain(fLow_, fHigh_, reL_, imL_, reR_, imR_, sr_, stftSize_);
            // Perform the upmix algorithm (compute center, left-side, right-side).
            doFreqUpmix(nBins);
            // Inverse FFT for left, right and center channels.
            for(unsigned int k = 0; k < nBins; k++){
                invL_.fdr(k) = reLs_[k];
                invL_.fdi(k) = imLs_[k];
            }
            invL_.ifft();
            for(unsigned int n = 0; n < stftSize_; n++){
                timeDomainL_[n] = invL_.td(n);
            }
            for(unsigned int k = 0; k < nBins; k++){
                invR_.fdr(k) = reRs_[k];
                invR_.fdi(k) = imRs_[k];
            }
            invR_.ifft();
            for(unsigned int n = 0; n < stftSize_; n++){
                timeDomainR_[n] = invR_.td(n);
            }
            for(unsigned int k = 0; k < nBins; k++){
                invL_.fdr(k) = reC_[k];
                invL_.fdi(k) = imC_[k];
            }
            invL_.ifft();
            for(unsigned int n = 0; n < stftSize_; n++){
                timeDomainC_[n] = invL_.td(n);
            }
            // Accumulate output using overlap-add.
            overlapAddL_.accumulate(timeDomainL_, synthesisWin_);
            overlapAddR_.accumulate(timeDomainR_, synthesisWin_);
            overlapAddC_.accumulate(timeDomainC_, synthesisWin_);
            static float chunkL[MAX_STFT_SIZE];
            static float chunkR[MAX_STFT_SIZE];
            static float chunkC[MAX_STFT_SIZE];
            overlapAddL_.popHop(hopSize_, chunkL);
            overlapAddR_.popHop(hopSize_, chunkR);
            overlapAddC_.popHop(hopSize_, chunkC);
            // Combine channels: left output = Ls + 0.5 * center, right output = Rs + 0.5 * center.
            for(unsigned int i = 0; i < hopSize_; i++){
                float valL = chunkL[i] + 0.5f * chunkC[i];
                float valR = chunkR[i] + 0.5f * chunkC[i];
                unsigned int outIndex = writePos + i;
                if(outIndex < hwBlockSize_){
                    outL[outIndex] = valL;
                    outR[outIndex] = valR;
                }
            }
            writePos += hopSize_;
        }
    }
private:
    // Perform the frequency-domain upmix processing.
    void doFreqUpmix(unsigned int nBins)
    {
        for(unsigned int k = 0; k < nBins; k++){
            float reCross = reL_[k] * reR_[k] + imL_[k] * imR_[k];
            float imCross = imL_[k] * reR_[k] - reL_[k] * imR_[k];
            float crossMag = std::sqrt(reCross * reCross + imCross * imCross);
            float magL = std::sqrt(reL_[k] * reL_[k] + imL_[k] * imL_[k]);
            float magR = std::sqrt(reR_[k] * reR_[k] + imR_[k] * imR_[k]);
            float denom = (magL * magR) + EPS;
            float coherence = crossMag / denom;
            float bal = (magL - magR) / (magL + magR + EPS);
            float cFactor = coherence * (1.f - std::fabs(bal));
            float reC = 0.5f * cFactor * (reL_[k] + reR_[k]);
            float imC = 0.5f * cFactor * (imL_[k] + imR_[k]);
            float reLs = reL_[k] - reC;
            float imLs = imL_[k] - imC;
            float reRs = reR_[k] - reC;
            float imRs = imR_[k] - imC;
            reC_[k]  = reC;    imC_[k]  = imC;
            reLs_[k] = reLs;   imLs_[k] = imLs;
            reRs_[k] = reRs;   imRs_[k] = imRs;
        }
    }
private:
    unsigned int hwBlockSize_ = 0; // Number of frames per hardware block
    float sr_ = 0.f;             // Sample rate
    float fLow_ = 0.f, fHigh_ = 22050.f; // Frequency range for this band
    float overlap_ = 0.75f;      // 75% overlap for WOLA
    unsigned int stftSize_ = 0;  // Internal STFT size for this band
    unsigned int hopSize_ = 0;   // Hop size = stftSize * (1 - overlap)
    // Frequency-domain temporary arrays
    float reL_[MAX_STFT_SIZE/2+1] = {0.f}, imL_[MAX_STFT_SIZE/2+1] = {0.f};
    float reR_[MAX_STFT_SIZE/2+1] = {0.f}, imR_[MAX_STFT_SIZE/2+1] = {0.f};
    float reC_[MAX_STFT_SIZE/2+1] = {0.f}, imC_[MAX_STFT_SIZE/2+1] = {0.f};
    float reLs_[MAX_STFT_SIZE/2+1] = {0.f}, imLs_[MAX_STFT_SIZE/2+1] = {0.f};
    float reRs_[MAX_STFT_SIZE/2+1] = {0.f}, imRs_[MAX_STFT_SIZE/2+1] = {0.f};
    // Time-domain temporary arrays
    float timeDomainL_[MAX_STFT_SIZE] = {0.f};
    float timeDomainR_[MAX_STFT_SIZE] = {0.f};
    float timeDomainC_[MAX_STFT_SIZE] = {0.f};
    // Analysis and synthesis windows
    float analysisWin_[MAX_STFT_SIZE] = {0.f};
    float synthesisWin_[MAX_STFT_SIZE] = {0.f};
    // Overlap-add accumulators
    OverlapAdd overlapAddL_, overlapAddR_, overlapAddC_;
    // Circular buffers for input
    CircularBuffer circBufL_, circBufR_;
    // FFT objects for left and right channels (both forward and inverse)
    Fft fwdL_, invL_;
    Fft fwdR_, invR_;
};

//--------------------------------------------------------------------------------
// MultiBandUpmix Aggregator Class
//--------------------------------------------------------------------------------
// This class manages multiple frequency bands. Each band is defined by a pair of
// adjacent band-edge frequencies. The STFT size for each band is computed dynamically
// using a Python-like logic: threshold = (sr * 8) / f_low, candidate = next_power_of_2(ceil(threshold)),
// then clamped so that candidate <= hwBlockSize * 4. This ensures that lower frequencies get
// higher resolution (larger block sizes) while higher frequencies use smaller blocks.
// The aggregator feeds the same input to all bands and sums the output from each band.
class MultiBandUpmix {
public:
    // Setup the aggregator.
    // Parameters:
    //   hwBlock: number of frames per hardware block.
    //   sr: sample rate.
    //   numBands: number of bands (should equal bandEdges array length - 1).
    //   bandEdges: array of band-edge frequencies (length = numBands + 1).
    void setup(unsigned int hwBlock, float sr, unsigned int numBands, const float *bandEdges)
    {
        sr_ = sr;
        hwBlockSize_ = hwBlock;
        numBands_ = numBands;
        if(numBands_ > MAX_BANDS)
            numBands_ = MAX_BANDS; // clamp to maximum available bands

        // Print a reference table for STFT size selection.
        printf("Reference Table: Low Frequency vs. STFT Size (max = hwBlock*4 = %u)\n", hwBlockSize_ * 4);
        printf("LowFreq (Hz)    Threshold      NextPow2   STFT Size\n");
        float sampleFreqs[] = {20.f, 40.f, 80.f, 160.f, 320.f, 640.f, 1280.f, 2560.f, 5120.f};
        for(unsigned int i = 0; i < sizeof(sampleFreqs) / sizeof(sampleFreqs[0]); i++){
            float f = sampleFreqs[i];
            float threshold = (sr_ * 8.f) / f;
            unsigned int np2 = nextPowerOf2((unsigned int)std::ceil(threshold));
            unsigned int stftCandidate = np2;
            if(stftCandidate > hwBlockSize_ * 4)
                stftCandidate = hwBlockSize_ * 4;
            printf("%8.1f       %8.1f       %8u    %8u\n", f, threshold, np2, stftCandidate);
        }
        printf("\n");

        // For each band, compute the STFT size and setup the band.
        for(unsigned int i = 0; i < numBands_; i++){
            float fLow = bandEdges[i];
            float fHigh = bandEdges[i + 1];
            unsigned int stftSize = computeBlockSizeForLowFreq(fLow, sr_, hwBlockSize_);
            printf("Band %u: fLow = %8.1f Hz, fHigh = %8.1f Hz --> STFT Size = %u\n", i, fLow, fHigh, stftSize);
            bands_[i].setup(hwBlockSize_, sr_, stftSize, fLow, fHigh);
        }
        printf("\n");
    }
    // Process the input: feed the input samples to each band and sum the outputs.
    // If a band does not yet have enough data, it contributes 0.
    void process(const float* inL, const float* inR, unsigned int frames,
                 float* outLeft, float* outRight)
    {
        // Clear output buffers.
        for(unsigned int i = 0; i < frames; i++){
            outLeft[i] = 0.f;
            outRight[i] = 0.f;
        }
        float tempL[4096], tempR[4096];
        // For each band, feed the input and, if ready, process one block and add its output.
        for(unsigned int i = 0; i < numBands_; i++){
            bands_[i].feed(inL, inR, frames);
            if(bands_[i].canProcessHwBlock()){
                bands_[i].doOneHwBlock(tempL, tempR);
                for(unsigned int n = 0; n < frames; n++){
                    outLeft[n] += tempL[n];
                    outRight[n] += tempR[n];
                }
            }
        }
    }
private:
    // Compute the STFT block size for a given low frequency using Python logic.
    // If f_low is <= 0, returns the maximum block size (hwBlock * 4).
    unsigned int computeBlockSizeForLowFreq(float f_low, float sr, unsigned int hwBlock) const {
        if(f_low <= 0.f)
            return hwBlock * 4;
        float threshold = (sr * 8.f) / f_low;
        unsigned int candidate = nextPowerOf2((unsigned int)std::ceil(threshold));
        if(candidate > hwBlock * 4)
            candidate = hwBlock * 4;
        return candidate;
    }
private:
    static constexpr unsigned int MAX_BANDS = 8;
    Overlap75UpmixBand bands_[MAX_BANDS];
    unsigned int numBands_ = 0;
    float sr_ = 0.f;
    unsigned int hwBlockSize_ = 0;
};

//--------------------------------------------------------------------------------
// Global variables and Bela entry points
//--------------------------------------------------------------------------------
static MultiBandUpmix gUpmix;

bool setup(BelaContext* context, void* userData)
{
    // Example: dynamically determine the number of bands from a static array.
    // The bandEdges array has length = numBands + 1.
    // Here we use a sample array of band edges. Adjust as needed.
    float bandEdges[] = {0.f, 400.f, 800.f, 3200.f, context->audioSampleRate * 0.5f};
    unsigned int numBands = sizeof(bandEdges) / sizeof(bandEdges[0]) - 1;
    gUpmix.setup(context->audioFrames, context->audioSampleRate, numBands, bandEdges);
    return true;
}

void render(BelaContext* context, void* userData)
{
    unsigned int frames = context->audioFrames;
    float inL[MAX_BUFFER_SIZE], inR[MAX_BUFFER_SIZE];
    // Read input audio samples.
    for(unsigned int n = 0; n < frames; n++){
        inL[n] = audioRead(context, n, 0);
        inR[n] = audioRead(context, n, 1);
    }
    float outL[MAX_BUFFER_SIZE], outR[MAX_BUFFER_SIZE];
    // Process the multi-band upmix.
    gUpmix.process(inL, inR, frames, outL, outR);
    // Write the output audio.
    for(unsigned int n = 0; n < frames; n++){
        audioWrite(context, n, 0, outL[n]);
        audioWrite(context, n, 1, outR[n]);
    }
}

void cleanup(BelaContext* context, void* userData)
{
    // No cleanup necessary.
}
