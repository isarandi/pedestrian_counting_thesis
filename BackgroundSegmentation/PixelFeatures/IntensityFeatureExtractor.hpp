#ifndef INTENSITYFEATUREEXTRACTOR_HPP
#define INTENSITYFEATUREEXTRACTOR_HPP

#include <BackgroundSegmentation/PixelFeatures/PixelFeatureExtractor.hpp>
#include <opencv2/core/core.hpp>
#include <stdx/cloning.hpp>
#include <memory>

namespace crowd { namespace bg {

class IntensityFeatureExtractor : public PixelFeatureExtractor
{
public:
    virtual auto getFeatures(cv::Mat const& img) const -> cv::Mat1d;

    CVX_CLONE_IN_DERIVED(IntensityFeatureExtractor)
};

}}

#endif // INTENSITYFEATUREEXTRACTOR_HPP
