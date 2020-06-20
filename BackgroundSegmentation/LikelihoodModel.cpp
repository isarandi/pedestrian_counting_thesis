#include <BackgroundSegmentation/LikelihoodModel.hpp>
#include <cvextra/math.hpp>
#include <opencv2/core/types_c.h>
#include <cmath>

using namespace std;
using namespace cv;
using namespace cvx;
using namespace crowd::bg;

LikelihoodModel::LikelihoodModel(double kernelSize, int maxWindowSize)
    : kernelSize(kernelSize)
    , maxWindowSize(maxWindowSize){}

void LikelihoodModel::update(Sample const& sample)
{
    if (samples.size() < maxWindowSize)
    {
        samples.push_back(sample);
    }
    else
    {
        // replace random sample if the window is full
        // this is somewhat like an exponential decay
        int randomIndex = std::uniform_int_distribution<int>(0, maxWindowSize-1)(randEng);
        samples[randomIndex] = sample;
    }
}

double LikelihoodModel::getLikelihood(Sample const& newSample) const
{
    double sumKernelResponses = 0;

    int nDim = newSample.size();

    for (auto& storedSample : samples)
    {
        double sumSquaredDiff = 0.0;
        for (int i = 0; i < nDim; ++i)
        {
            sumSquaredDiff += cvx::sq(newSample[i] - storedSample[i]);
        }

        sumKernelResponses += std::exp((float)(-sumSquaredDiff/(2*kernelSize)));
    }

    return sumKernelResponses/samples.size() * std::pow(CV_PI*kernelSize, -nDim*0.5);
}

