#ifndef NEIGHBORHOODBACKGROUNDMODEL_HPP
#define NEIGHBORHOODBACKGROUNDMODEL_HPP

#include <BackgroundSegmentation/BackgroundModel.hpp>
#include <BackgroundSegmentation/LikelihoodModel.hpp>
#include <BackgroundSegmentation/PixelFeatures/PixelFeatureExtractor.hpp>
#include <opencv2/core/core.hpp>
#include <stdx/cloning.hpp>
#include <memory>
#include <random>
#include <vector>

namespace crowd { namespace bg {

class NeighborhoodBackgroundModel : public BackgroundModel
{
public:
    NeighborhoodBackgroundModel(
            double threshold,
            int blindInitLength,
            double selectiveUpdateThreshold,
            PixelFeatureExtractor const& extractor,
            LikelihoodModel const& pixelModelPrototype,
            double neighborhoodRadius);

    virtual void segmentNext(cv::InputArray image, cv::OutputArray fgmask);
    virtual void initialize(cv::Size processingSize);

    auto getPixelModels() const -> std::vector<LikelihoodModel> const& {return pixelModels;}

    CVX_CLONE_IN_DERIVED(NeighborhoodBackgroundModel)

private:
    cv::Size frameSize;

    double threshold;
    double selectiveUpdateThreshold;

    stdx::cloned_unique_ptr<PixelFeatureExtractor> featureExtractor;

    LikelihoodModel pixelModelPrototype;
    std::vector<LikelihoodModel> pixelModels;

    double neighborhoodRadius;
    std::vector<cv::Vec2i> neighborhoodDeltas;

    int blindInitLength;
    int iFrame;
    std::default_random_engine rand;
};

}}

#endif // NEIGHBORHOODBACKGROUNDMODEL_HPP
