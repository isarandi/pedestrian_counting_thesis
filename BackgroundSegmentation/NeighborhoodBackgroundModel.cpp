#include <BackgroundSegmentation/LikelihoodModel.hpp>
#include <BackgroundSegmentation/NeighborhoodBackgroundModel.hpp>
#include <BackgroundSegmentation/PixelFeatures/PixelFeatureExtractor.hpp>
#include <cvextra/coords.hpp>
#include <cvextra/core.hpp>
#include <cvextra/mats.hpp>
#include <opencv2/core/mat.hpp>
#include <opencv2/core/core.hpp>
#include <cmath>
#include <random>

using namespace std;
using namespace cv;
using namespace cvx;
using namespace crowd::bg;

NeighborhoodBackgroundModel::NeighborhoodBackgroundModel(
        double threshold,
        int blindInitLength,
        double selectiveUpdateThreshold,
        PixelFeatureExtractor const& extractor,
        LikelihoodModel const& pixelModelPrototype,
        double neighborhoodRadius)
    : threshold(threshold)
    , blindInitLength(blindInitLength)
    , selectiveUpdateThreshold(selectiveUpdateThreshold)
    , featureExtractor(extractor.clone())
    , pixelModelPrototype(pixelModelPrototype)
    , neighborhoodRadius(neighborhoodRadius)
{
    int bound = std::ceil(neighborhoodRadius);

    for (int y = -bound; y < bound; ++y)
    {
        for (int x = -bound; x < bound; ++x)
        {
            Vec2i diff(x,y);
            if (cv::norm(diff) <= neighborhoodRadius)
            {
                neighborhoodDeltas.push_back(diff);
            }
        }
    }
}

void NeighborhoodBackgroundModel::initialize(Size processingSize)
{
    frameSize = processingSize;
    pixelModels.clear();

    for (int i = 0; i < frameSize.area(); ++i)
    {
        pixelModels.push_back(pixelModelPrototype);
    }

    iFrame=0;
}

void NeighborhoodBackgroundModel::segmentNext(InputArray _nextImage, OutputArray _fgmask)
{
    Mat nextImage = _nextImage.getMat();
    Mat1d features = featureExtractor->getFeatures(nextImage);

    // compute max likelihoods over the neighborhoods of each pixel
    Mat1d maxLikelihoods(frameSize, 0);

    #pragma omp parallel for
    for (int iCenter = 0; iCenter < frameSize.area() ; ++iCenter)
    {
        Point center = {iCenter % frameSize.width, iCenter / frameSize.width};
        vector<double> centerPixelFeatures = features.row(iCenter);

        double maxLikelihood = 0;

        for (auto const& neighborhoodDelta : neighborhoodDeltas)
        {
            Point neighbor = center + neighborhoodDelta;

            if (cvx::contains(frameSize, neighbor))
            {
                int iNeigh = neighbor.y * frameSize.width + neighbor.x;

                double likelihoodAtNeighbor = pixelModels[iNeigh].getLikelihood(centerPixelFeatures);
                if (likelihoodAtNeighbor > maxLikelihood)
                {
                    maxLikelihood = likelihoodAtNeighbor;
                }
            }
        }

        maxLikelihoods(center) = maxLikelihood;
    }

    // update pixel models

    // we do blind update in the beginning and at random occasions
    bool const blindUpdate = iFrame < blindInitLength || std::bernoulli_distribution(0.5)(rand);

    #pragma omp parallel for
    for (int iCenter = 0; iCenter < frameSize.area() ; ++iCenter)
    {
        if (blindUpdate
                || at(maxLikelihoods, iCenter) > selectiveUpdateThreshold)
        {
            pixelModels[iCenter].update(features.row(iCenter));
        }
    }

    cv::compare(maxLikelihoods, threshold, _fgmask, CMP_LT);

    // step the frame count used for blind init
    ++iFrame;
}
