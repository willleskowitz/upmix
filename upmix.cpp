#include <Bela.h>
#include <libraries/Fft/Fft.h>
#include <cmath>
#include <stdexcept>
#include <algorithm>

// Tune these limits as needed
static constexpr unsigned int MAX_STFT_SIZE  = 8192;
static constexpr unsigned int MAX_QUEUE_SIZE = 4 * MAX_STFT_SIZE;

static constexpr float EPS = 1e-12f;

// Generate a Blackman–Harris window in-place
static void makeBlackmanHarris(float* w, unsigned int size)
{
    constexpr float a0 = 0.35875f;
    constexpr float a1 = 0.48829f;
    constexpr float a2 = 0.14128f;
    constexpr float a3 = 0.01168f;
    for(unsigned int n=0; n < size; n++){
        float ratio = (float)n / (float)(size - 1);
        w[n] = a0
             - a1 * cosf(2.f * M_PI * ratio)
             + a2 * cosf(4.f * M_PI * ratio)
             - a3 * cosf(6.f * M_PI * ratio);
    }
}

// Compute a WOLA‐corrected synthesis window from an analysis window
// for 75% overlap. For each sample n, divides wA[n] by sum of squares
// of all overlapping wA[] segments.
static void makeWolaSynthesisWindow(float* synth, const float* analysis,
                                    unsigned int size, float overlap)
{
    unsigned int hop = (unsigned int)std::lround(size * (1.f - overlap));
    if(hop < 1)
        throw std::runtime_error("Overlap too large; hop < 1");

    // e.g. 4 if overlap=0.75
    unsigned int K = (unsigned int)std::lround(1.0f / (1.0f - overlap));

    for(unsigned int n = 0; n < size; n++){
        float sumSq = 0.f;
        // sum over each overlapping segment
        for(unsigned int k = 0; k < K; k++){
            unsigned int idx = (n + k * hop) % size;
            sumSq += analysis[idx] * analysis[idx];
        }
        synth[n] = analysis[n] / (sumSq + EPS);
    }
}

//--------------------------------------------------------------------------------
// Fixed-size CircularBuffer
//--------------------------------------------------------------------------------
class CircularBuffer {
public:
    void setup(unsigned int size) {
        if(size > MAX_STFT_SIZE)
            throw std::runtime_error("CircularBuffer size too large");
        size_ = size;
        writePos_  = 0;
        fillCount_ = 0;
        for(unsigned int i=0; i<size_; i++){
            buffer_[i] = 0.f;
        }
    }

    void writeSamples(const float* in, unsigned int numSamples) {
        for(unsigned int i=0; i < numSamples; i++){
            buffer_[writePos_] = in[i];
            writePos_ = (writePos_ + 1) % size_;
        }
        fillCount_ += numSamples;
        // clamp fillCount_ to size_
        if(fillCount_ > size_)
            fillCount_ = size_;
    }

    bool canProcess(unsigned int hopSize) const {
        return (fillCount_ >= hopSize);
    }

    void readBlock(float* out, unsigned int stftSize, unsigned int hopSize) {
        if(stftSize > size_)
            throw std::runtime_error("STFT size too large for CircularBuffer");
        unsigned int readPos = (writePos_ + size_ - stftSize) % size_;
        for(unsigned int n=0; n < stftSize; n++){
            out[n] = buffer_[(readPos + n) % size_];
        }
        if(fillCount_ >= hopSize)
            fillCount_ -= hopSize;
        else
            fillCount_ = 0;
    }

private:
    float buffer_[MAX_STFT_SIZE];
    unsigned int size_ = 0;
    unsigned int writePos_ = 0;
    unsigned int fillCount_ = 0;
};

//--------------------------------------------------------------------------------
// Overlap-Add class
//--------------------------------------------------------------------------------
class OverlapAdd {
public:
    void setup(unsigned int size) {
        if(size > MAX_STFT_SIZE)
            throw std::runtime_error("OverlapAdd size too large");
        size_ = size;
        for(unsigned int i=0; i<size_; i++){
            accum_[i] = 0.f;
        }
    }

    void accumulate(const float* block, const float* window) {
        for(unsigned int n=0; n < size_; n++){
            accum_[n] += block[n] * window[n];
        }
    }

    // Pop hopSize from accum_, shifting the rest
    void popHop(unsigned int hopSize, float* out, unsigned int& outCount) {
        for(unsigned int n=0; n < hopSize; n++){
            out[outCount + n] = accum_[n];
        }
        for(unsigned int n=hopSize; n < size_; n++){
            accum_[n - hopSize] = accum_[n];
        }
        for(unsigned int n=size_ - hopSize; n < size_; n++){
            accum_[n] = 0.f;
        }
        outCount += hopSize;
    }

private:
    float accum_[MAX_STFT_SIZE];
    unsigned int size_ = 0;
};

//--------------------------------------------------------------------------------
// L/C/R upmix w/ 75% overlap and WOLA synthesis window
//--------------------------------------------------------------------------------
class Overlap75Upmix {
public:
    void setup(unsigned int hwBlockSize, float sampleRate)
    {
        hwBlockSize_ = hwBlockSize;
        sr_          = sampleRate;
        overlapFactor_ = 0.75f;

        stftSize_ = hwBlockSize_ * 4;
        if(stftSize_ > MAX_STFT_SIZE)
            throw std::runtime_error("Requested stftSize > MAX_STFT_SIZE");

        hopSize_ = (unsigned int)std::lround(stftSize_ * (1.f - overlapFactor_));
        nBins_   = stftSize_ / 2 + 1;

        // FFT objects
        if(fwdL_.setup(stftSize_) != 0 || invL_.setup(stftSize_) != 0)
            throw std::runtime_error("FFT setup (Left) failed");
        if(fwdR_.setup(stftSize_) != 0 || invR_.setup(stftSize_) != 0)
            throw std::runtime_error("FFT setup (Right) failed");
        if(invC_.setup(stftSize_) != 0)
            throw std::runtime_error("FFT setup (Center) failed");

        // Make analysis window (Blackman-Harris)
        makeBlackmanHarris(analysisWin_, stftSize_);
        // Make a WOLA-corrected synthesis window
        makeWolaSynthesisWindow(synthesisWin_, analysisWin_, stftSize_, overlapFactor_);

        // Zero arrays
        for(unsigned int i=0; i<MAX_STFT_SIZE; i++){
            timeDomainL_[i] = 0.f;
            timeDomainR_[i] = 0.f;
            timeDomainC_[i] = 0.f;
        }
        outQueueLeftLen_  = 0;
        outQueueRightLen_ = 0;
        for(unsigned int i=0; i<MAX_QUEUE_SIZE; i++){
            outQueueLeft_[i]  = 0.f;
            outQueueRight_[i] = 0.f;
        }

        for(unsigned int i=0; i<nBins_; i++){
            reL_[i]  = imL_[i]  = 0.f;
            reR_[i]  = imR_[i]  = 0.f;
            reC_[i]  = imC_[i]  = 0.f;
            reLs_[i] = imLs_[i] = 0.f;
            reRs_[i] = imRs_[i] = 0.f;
        }

        // OverlapAdd accumulators
        overlapAddL_.setup(stftSize_);
        overlapAddR_.setup(stftSize_);
        overlapAddC_.setup(stftSize_);

        // Ring buffers
        circBufL_.setup(stftSize_);
        circBufR_.setup(stftSize_);
    }

    void process(const float* inL, const float* inR, unsigned int frames,
                 float* outLeft, float* outRight)
    {
        // Zero outputs
        for(unsigned int i=0; i<frames; i++){
            outLeft[i]  = 0.f;
            outRight[i] = 0.f;
        }

        // Feed ring buffers
        circBufL_.writeSamples(inL, frames);
        circBufR_.writeSamples(inR, frames);

        // Run STFT blocks if possible
        while(circBufL_.canProcess(hopSize_) && circBufR_.canProcess(hopSize_)){
            doOneStftBlock();
        }

        // Pop from outQueue
        unsigned int toPop = std::min(outQueueLeftLen_, frames);
        toPop = std::min(toPop, outQueueRightLen_);
        for(unsigned int i=0; i<toPop; i++){
            outLeft[i]  = outQueueLeft_[i];
            outRight[i] = outQueueRight_[i];
        }
        // Shift leftover
        if(toPop > 0){
            unsigned int leftoverL = outQueueLeftLen_ - toPop;
            unsigned int leftoverR = outQueueRightLen_ - toPop;
            for(unsigned int i=0; i<leftoverL; i++){
                outQueueLeft_[i] = outQueueLeft_[i + toPop];
            }
            for(unsigned int i=0; i<leftoverR; i++){
                outQueueRight_[i] = outQueueRight_[i + toPop];
            }
            outQueueLeftLen_  = leftoverL;
            outQueueRightLen_ = leftoverR;
        }
    }

private:
    void doOneStftBlock()
    {
        // A) read stftSize from ring buffers
        circBufL_.readBlock(timeDomainL_, stftSize_, hopSize_);
        circBufR_.readBlock(timeDomainR_, stftSize_, hopSize_);

        // B) forward FFT
        for(unsigned int n=0; n<stftSize_; n++){
            fwdL_.td(n) = timeDomainL_[n] * analysisWin_[n];
        }
        fwdL_.fft();
        for(unsigned int k=0; k<nBins_; k++){
            reL_[k] = fwdL_.fdr(k);
            imL_[k] = fwdL_.fdi(k);
        }

        for(unsigned int n=0; n<stftSize_; n++){
            fwdR_.td(n) = timeDomainR_[n] * analysisWin_[n];
        }
        fwdR_.fft();
        for(unsigned int k=0; k<nBins_; k++){
            reR_[k] = fwdR_.fdr(k);
            imR_[k] = fwdR_.fdi(k);
        }

        // C) L/C/R upmix in freq domain
        for(unsigned int k=0; k<nBins_; k++){
            float reCross  = reL_[k]*reR_[k] + imL_[k]*imR_[k];
            float imCross  = imL_[k]*reR_[k] - reL_[k]*imR_[k];
            float crossMag = std::sqrt(reCross*reCross + imCross*imCross);

            float magL = std::sqrt(reL_[k]*reL_[k] + imL_[k]*imL_[k]);
            float magR = std::sqrt(reR_[k]*reR_[k] + imR_[k]*imR_[k]);
            float denom     = (magL*magR) + EPS;
            float coherence = crossMag / denom;
            float bal       = (magL - magR)/(magL + magR + EPS);
            float cFactor   = coherence * (1.f - std::fabs(bal));

            // Center
            reC_[k]  = 0.5f * cFactor * (reL_[k] + reR_[k]);
            imC_[k]  = 0.5f * cFactor * (imL_[k] + imR_[k]);
            // Ls, Rs
            reLs_[k] = reL_[k] - reC_[k];
            imLs_[k] = imL_[k] - imC_[k];
            reRs_[k] = reR_[k] - reC_[k];
            imRs_[k] = imR_[k] - imC_[k];
        }

        // D) iFFT => Ls
        for(unsigned int k=0; k<nBins_; k++){
            invL_.fdr(k) = reLs_[k];
            invL_.fdi(k) = imLs_[k];
        }
        invL_.ifft();
        for(unsigned int n=0; n<stftSize_; n++){
            timeDomainL_[n] = invL_.td(n);
        }

        // => Rs
        for(unsigned int k=0; k<nBins_; k++){
            invR_.fdr(k) = reRs_[k];
            invR_.fdi(k) = imRs_[k];
        }
        invR_.ifft();
        for(unsigned int n=0; n<stftSize_; n++){
            timeDomainR_[n] = invR_.td(n);
        }

        // => Center
        for(unsigned int k=0; k<nBins_; k++){
            invC_.fdr(k) = reC_[k];
            invC_.fdi(k) = imC_[k];
        }
        invC_.ifft();
        for(unsigned int n=0; n<stftSize_; n++){
            timeDomainC_[n] = invC_.td(n);
        }

        // E) Overlap-add
        overlapAddL_.accumulate(timeDomainL_, synthesisWin_);
        overlapAddR_.accumulate(timeDomainR_, synthesisWin_);
        overlapAddC_.accumulate(timeDomainC_, synthesisWin_);

        // F) pop hop frames
        unsigned int chunkLenL = 0, chunkLenR = 0, chunkLenC = 0;
        overlapAddL_.popHop(hopSize_, chunkL_, chunkLenL);
        overlapAddR_.popHop(hopSize_, chunkR_, chunkLenR);
        overlapAddC_.popHop(hopSize_, chunkC_, chunkLenC);

        // G) Combine => outLeft = Ls + 0.5*C, outRight = Rs + 0.5*C
        for(unsigned int i=0; i<chunkLenL; i++){
            float outL = chunkL_[i] + 0.5f * chunkC_[i];
            float outR = chunkR_[i] + 0.5f * chunkC_[i];
            if(outQueueLeftLen_ < MAX_QUEUE_SIZE){
                outQueueLeft_[outQueueLeftLen_++] = outL;
            }
            if(outQueueRightLen_ < MAX_QUEUE_SIZE){
                outQueueRight_[outQueueRightLen_++] = outR;
            }
        }
    }

private:
    unsigned int hwBlockSize_ = 0;
    unsigned int stftSize_    = 0;
    unsigned int hopSize_     = 0;
    unsigned int nBins_       = 0;
    float overlapFactor_       = 0.75f;
    float sr_                  = 0.f;

    // Windows
    float analysisWin_[MAX_STFT_SIZE];
    float synthesisWin_[MAX_STFT_SIZE];

    // Time‐domain
    float timeDomainL_[MAX_STFT_SIZE];
    float timeDomainR_[MAX_STFT_SIZE];
    float timeDomainC_[MAX_STFT_SIZE];

    // Freq domain: L, R, C
    float reL_[MAX_STFT_SIZE/2 + 1], imL_[MAX_STFT_SIZE/2 + 1];
    float reR_[MAX_STFT_SIZE/2 + 1], imR_[MAX_STFT_SIZE/2 + 1];
    float reC_[MAX_STFT_SIZE/2 + 1], imC_[MAX_STFT_SIZE/2 + 1];

    // Ls/Rs
    float reLs_[MAX_STFT_SIZE/2 + 1], imLs_[MAX_STFT_SIZE/2 + 1];
    float reRs_[MAX_STFT_SIZE/2 + 1], imRs_[MAX_STFT_SIZE/2 + 1];

    // OverlapAdd accumulators
    OverlapAdd overlapAddL_, overlapAddR_, overlapAddC_;

    // Ring buffers
    class CircularBuffer circBufL_, circBufR_;

    // Output queues
    float outQueueLeft_[MAX_QUEUE_SIZE];
    float outQueueRight_[MAX_QUEUE_SIZE];
    unsigned int outQueueLeftLen_  = 0;
    unsigned int outQueueRightLen_ = 0;

    // Temp chunk arrays
    float chunkL_[MAX_STFT_SIZE];
    float chunkR_[MAX_STFT_SIZE];
    float chunkC_[MAX_STFT_SIZE];

    // FFT objects
    Fft fwdL_, invL_;
    Fft fwdR_, invR_;
    Fft invC_;
};

static Overlap75Upmix gTest;

bool setup(BelaContext* context, void* userData)
{
    gTest.setup(context->audioFrames, context->audioSampleRate);
    return true;
}

void render(BelaContext* context, void* userData)
{
    unsigned int frames = context->audioFrames;
    float inL [MAX_STFT_SIZE], inR [MAX_STFT_SIZE];
    float outL[MAX_STFT_SIZE], outR[MAX_STFT_SIZE];

    // Read input
    for(unsigned int n=0; n < frames; n++){
        inL[n] = audioRead(context, n, 0);
        inR[n] = audioRead(context, n, 1);
    }

    // Process
    gTest.process(inL, inR, frames, outL, outR);

    // Write output
    for(unsigned int n=0; n < frames; n++){
        audioWrite(context, n, 0, outL[n]);
        audioWrite(context, n, 1, outR[n]);
    }
}

void cleanup(BelaContext* context, void* userData)
{
    // Nothing
}
