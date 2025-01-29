#include <Bela.h>
#include <libraries/Fft/Fft.h>
#include <cmath>
#include <vector>
#include <stdexcept>
#include <algorithm>

// A small epsilon to avoid dividing by zero
static constexpr float EPS = 1e-12f;

// Generate a Blackman–Harris window of length 'size'
static std::vector<float> makeBlackmanHarris(unsigned int size)
{
    std::vector<float> w(size);
    constexpr float a0 = 0.35875f;
    constexpr float a1 = 0.48829f;
    constexpr float a2 = 0.14128f;
    constexpr float a3 = 0.01168f;
    for(unsigned int n = 0; n < size; n++){
        float ratio = (float)n / (float)(size - 1);
        w[n] = a0
             - a1 * cosf(2.f * M_PI * ratio)
             + a2 * cosf(4.f * M_PI * ratio)
             - a3 * cosf(6.f * M_PI * ratio);
    }
    return w;
}

// CircularBuffer for a single channel
class CircularBuffer {
public:
    void setup(unsigned int size)
    {
        size_ = size;
        buffer_.resize(size_, 0.f);
        writePos_  = 0;
        fillCount_ = 0;
    }

    void writeSamples(const float* in, unsigned int numSamples)
    {
        for(unsigned int i = 0; i < numSamples; i++){
            buffer_[writePos_] = in[i];
            writePos_ = (writePos_ + 1) % size_;
        }
        fillCount_ += numSamples;
    }

    bool canProcess(unsigned int hopSize) const
    {
        return (fillCount_ >= hopSize);
    }

    // Read 'stftSize' samples, then consume only 'hopSize'
    void readBlock(std::vector<float>& out, unsigned int stftSize, unsigned int hopSize)
    {
        if(stftSize > size_)
            throw std::runtime_error("STFT size larger than circular buffer");

        unsigned int readPos = (writePos_ + size_ - stftSize) % size_;
        for(unsigned int n = 0; n < stftSize; n++){
            out[n] = buffer_[(readPos + n) % size_];
        }

        fillCount_ -= hopSize;
        if((int)fillCount_ < 0)
            fillCount_ = 0;
    }

private:
    std::vector<float> buffer_;
    unsigned int size_      = 0;
    unsigned int writePos_  = 0;
    unsigned int fillCount_ = 0;
};

// Weighted Overlap–Add for a single channel
class OverlapAdd {
public:
    void setup(unsigned int size)
    {
        size_ = size;
        accum_.resize(size_, 0.f);
    }

    void accumulate(const std::vector<float>& block, const std::vector<float>& window)
    {
        for(unsigned int n = 0; n < size_; n++){
            accum_[n] += block[n] * window[n];
        }
    }

    // Pop the newest 'hopSize' samples from 'accum_', shifting the rest
    void popHop(unsigned int hopSize, std::vector<float>& outQueue)
    {
        for(unsigned int n = 0; n < hopSize; n++){
            outQueue.push_back(accum_[n]);
        }
        for(unsigned int n = hopSize; n < size_; n++){
            accum_[n - hopSize] = accum_[n];
        }
        for(unsigned int n = size_ - hopSize; n < size_; n++){
            accum_[n] = 0.f;
        }
    }

private:
    std::vector<float> accum_;
    unsigned int size_ = 0;
};

//--------------------------------------------------------------
// Single-band L/C/R Upmix at 75% overlap
//--------------------------------------------------------------
class Overlap75Upmix {
public:
    void setup(unsigned int hardwareBlock, float sampleRate)
    {
        hwBlockSize_ = hardwareBlock;
        sr_          = sampleRate;
        overlapFactor_ = 0.75f;

        // For 75% overlap, we set stftSize to 4× the hardware block size
        stftSize_ = hwBlockSize_ * 4;
        hopSize_  = static_cast<unsigned int>(std::lround(stftSize_ * (1.f - overlapFactor_)));

        // Forward & inverse FFT setups for Left & Right
        if(fwdL_.setup(stftSize_) != 0 || invL_.setup(stftSize_) != 0)
            throw std::runtime_error("FFT setup (Left) failed");
        if(fwdR_.setup(stftSize_) != 0 || invR_.setup(stftSize_) != 0)
            throw std::runtime_error("FFT setup (Right) failed");

        // Windows
        analysisWin_  = makeBlackmanHarris(stftSize_);
        synthesisWin_ = makeBlackmanHarris(stftSize_);

        // Frequency-domain buffers
        size_t nBins = stftSize_ / 2 + 1;
        reL_.resize(nBins, 0.f); imL_.resize(nBins, 0.f);
        reR_.resize(nBins, 0.f); imR_.resize(nBins, 0.f);
        reC_.resize(nBins, 0.f); imC_.resize(nBins, 0.f);
        reLs_.resize(nBins,0.f); imLs_.resize(nBins,0.f);
        reRs_.resize(nBins,0.f); imRs_.resize(nBins,0.f);

        // Time-domain blocks
        timeDomainL_.resize(stftSize_, 0.f);
        timeDomainR_.resize(stftSize_, 0.f);
        timeDomainC_.resize(stftSize_, 0.f);

        // OverlapAdd accumulators
        overlapAddL_.setup(stftSize_);
        overlapAddR_.setup(stftSize_);
        overlapAddC_.setup(stftSize_);

        // Ring buffers for Left & Right
        circBufL_.setup(stftSize_);
        circBufR_.setup(stftSize_);

        // Final output queues
        outQueueLeft_.clear();
        outQueueRight_.clear();
    }

    // Real-time process: feed L & R, get upmixed L & R
    void process(const float* inL, const float* inR,
                 unsigned int frames,
                 std::vector<float>& outLeft,
                 std::vector<float>& outRight)
    {
        outLeft.assign(frames, 0.f);
        outRight.assign(frames, 0.f);

        // 1) Write new samples to ring buffers
        circBufL_.writeSamples(inL, frames);
        circBufR_.writeSamples(inR, frames);

        // 2) Process STFT blocks whenever hopSize is reached
        while(circBufL_.canProcess(hopSize_) && circBufR_.canProcess(hopSize_)) {
            doOneStftBlock();
        }

        // 3) Pop final audio from outQueueLeft_/outQueueRight_
        unsigned int toPop = std::min((unsigned int)outQueueLeft_.size(), frames);
        toPop = std::min(toPop, (unsigned int)outQueueRight_.size());
        for(unsigned int i = 0; i < toPop; i++){
            outLeft[i]  = outQueueLeft_[i];
            outRight[i] = outQueueRight_[i];
        }
        if(toPop > 0){
            outQueueLeft_.erase(outQueueLeft_.begin(), outQueueLeft_.begin() + toPop);
            outQueueRight_.erase(outQueueRight_.begin(), outQueueRight_.begin() + toPop);
        }
    }

private:
    void doOneStftBlock()
    {
        // A) Read stftSize_ from ring buffers (Left & Right)
        circBufL_.readBlock(timeDomainL_, stftSize_, hopSize_);
        circBufR_.readBlock(timeDomainR_, stftSize_, hopSize_);

        // B) Forward FFT on each channel
        for(unsigned int n=0; n<stftSize_; n++){
            fwdL_.td(n) = timeDomainL_[n] * analysisWin_[n];
        }
        fwdL_.fft();
        copyFreqDomain(fwdL_, reL_, imL_);

        for(unsigned int n=0; n<stftSize_; n++){
            fwdR_.td(n) = timeDomainR_[n] * analysisWin_[n];
        }
        fwdR_.fft();
        copyFreqDomain(fwdR_, reR_, imR_);

        // C) Upmix in frequency domain => center & side
        upmixFreqDomain();

        // D) iFFT => time domain for Ls, Rs, C
        invL_.ifft(reLs_, imLs_);
        for(unsigned int n=0; n<stftSize_; n++){
            timeDomainL_[n] = invL_.td(n);
        }

        invR_.ifft(reRs_, imRs_);
        for(unsigned int n=0; n<stftSize_; n++){
            timeDomainR_[n] = invR_.td(n);
        }

        invL_.ifft(reC_, imC_);
        for(unsigned int n=0; n<stftSize_; n++){
            timeDomainC_[n] = invL_.td(n);
        }

        // E) Overlap-Add for L, R, C
        overlapAddL_.accumulate(timeDomainL_, synthesisWin_);
        overlapAddR_.accumulate(timeDomainR_, synthesisWin_);
        overlapAddC_.accumulate(timeDomainC_, synthesisWin_);

        // F) Pop out the new hop
        std::vector<float> chunkL, chunkR, chunkC;
        overlapAddL_.popHop(hopSize_, chunkL);
        overlapAddR_.popHop(hopSize_, chunkR);
        overlapAddC_.popHop(hopSize_, chunkC);

        // G) Combine => outLeft = Ls + 0.5*C; outRight = Rs + 0.5*C
        for(unsigned int i=0; i<chunkL.size(); i++){
            outQueueLeft_.push_back(chunkL[i] + 0.5f * chunkC[i]);
            outQueueRight_.push_back(chunkR[i] + 0.5f * chunkC[i]);
        }
    }

    // Copy real/imag data from an Fft object into arrays
    void copyFreqDomain(Fft& fftObj, std::vector<float>& re, std::vector<float>& im)
    {
        unsigned int nBins = stftSize_ / 2 + 1;
        for(unsigned int k=0; k<nBins; k++){
            re[k] = fftObj.fdr(k);
            im[k] = fftObj.fdi(k);
        }
    }

    // Compute single-band L/C/R in frequency domain
    void upmixFreqDomain()
    {
        // cross = L * conj(R)
        // cross_mag = |cross|
        // magL=|L|, magR=|R|
        // coherence = cross_mag/(magL*magR+EPS)
        // balance=(magL-magR)/(magL+magR+EPS)
        // centerFactor= coherence*(1-abs(balance))
        // specC= 0.5 * centerFactor*(L+R)
        // Ls= L- specC
        // Rs= R- specC
        size_t nBins = reL_.size();
        for(size_t k=0; k<nBins; k++){
            float reCross = (reL_[k]*reR_[k]) + (imL_[k]*imR_[k]);
            float imCross = (imL_[k]*reR_[k]) - (reL_[k]*imR_[k]);
            float crossMag= std::sqrt(reCross*reCross + imCross*imCross);

            float magL= std::sqrt(reL_[k]*reL_[k] + imL_[k]*imL_[k]);
            float magR= std::sqrt(reR_[k]*reR_[k] + imR_[k]*imR_[k]);
            float denom = (magL*magR) + EPS;
            float coherence= crossMag/ denom;
            float bal = (magL - magR)/(magL + magR + EPS);
            float cFactor= coherence*(1.f - std::fabs(bal));

            float reC= 0.5f*cFactor*(reL_[k] + reR_[k]);
            float imC= 0.5f*cFactor*(imL_[k] + imR_[k]);

            float reLs= reL_[k] - reC;
            float imLs= imL_[k] - imC;
            float reRs= reR_[k] - reC;
            float imRs= imR_[k] - imC;

            reC_[k]  = reC;   imC_[k]  = imC;
            reLs_[k] = reLs;  imLs_[k] = imLs;
            reRs_[k] = reRs;  imRs_[k] = imRs;
        }
    }

private:
    // Bela config
    unsigned int hwBlockSize_ = 0;
    unsigned int stftSize_    = 0;
    float overlapFactor_       = 0.f;
    unsigned int hopSize_      = 0;
    float sr_                  = 0.f;

    // FFT objects for L & R
    Fft fwdL_, invL_;
    Fft fwdR_, invR_;

    // freq domain for L & R
    std::vector<float> reL_, imL_;
    std::vector<float> reR_, imR_;

    // freq domain for center & sides
    std::vector<float> reC_, imC_;
    std::vector<float> reLs_, imLs_;
    std::vector<float> reRs_, imRs_;

    // time domain
    std::vector<float> timeDomainL_, timeDomainR_, timeDomainC_;

    // windows
    std::vector<float> analysisWin_, synthesisWin_;

    // Overlap-Add for each channel
    OverlapAdd overlapAddL_, overlapAddR_, overlapAddC_;

    // ring buffers for Left & Right
    CircularBuffer circBufL_, circBufR_;

    // final output queues
    std::vector<float> outQueueLeft_, outQueueRight_;
};

// Global instance
static Overlap75Upmix gTest;

// Configure & start up
bool setup(BelaContext* context, void* userData)
{
    gTest.setup(context->audioFrames, context->audioSampleRate);
    return true;
}

// Audio callback
void render(BelaContext* context, void* userData)
{
    unsigned int frames = context->audioFrames;

    // Read stereo from channels 0 & 1
    std::vector<float> inL(frames), inR(frames);
    for(unsigned int n = 0; n < frames; n++){
        inL[n] = audioRead(context, n, 0);
        inR[n] = audioRead(context, n, 1);
    }

    // Do upmix
    std::vector<float> outLeft, outRight;
    gTest.process(inL.data(), inR.data(), frames, outLeft, outRight);

    // Write final L & R
    for(unsigned int n=0; n < frames; n++){
        float L = (n < outLeft.size()) ? outLeft[n] : 0.f;
        float R = (n < outRight.size())? outRight[n]:0.f;
        audioWrite(context, n, 0, L);
        audioWrite(context, n, 1, R);
    }
}

void cleanup(BelaContext* context, void* userData)
{
    // no cleanup
}
