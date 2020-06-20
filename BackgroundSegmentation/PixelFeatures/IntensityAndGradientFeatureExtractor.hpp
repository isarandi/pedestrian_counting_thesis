#ifndef INTENSITYANDGRADIENTFEATUREEXTRACTOR_HPP
#define INTENSITYANDGRADIENTFEATUREEXTRACTOR_HPP

#include <BackgroundSegmentation/PixelFeatures/PixelFeatureExtractor.hpp>
#include <opencv2/core/core.hpp>
#include <stdx/cloning.hpp>
#include <memory>

namespace crowd { namespace bg {

class IntensityAndGradientFeatureExtractor : public PixelFeatureExtractor
{
public:
	IntensityAndGradientFeatureExtractor(double gradientScale):gradientScale(gradientScale){}

    virtual auto getFeatures(cv::Mat const& img) const -> cv::Mat1d;

    CVX_CLONE_IN_DERIVED(IntensityAndGradientFeatureExtractor)
private:
    double gradientScale;
};

}}

#endif // INTENSITYANDGRADIENTFEATUREEXTRACTOR_HPP
