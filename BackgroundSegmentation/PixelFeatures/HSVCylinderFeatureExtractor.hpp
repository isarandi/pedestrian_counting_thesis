#ifndef HSVCYLINDERFEATUREEXTRACTOR_HPP
#define HSVCYLINDERFEATUREEXTRACTOR_HPP

#include <BackgroundSegmentation/PixelFeatures/PixelFeatureExtractor.hpp>
#include <opencv2/core/core.hpp>
#include <stdx/cloning.hpp>
#include <memory>

namespace crowd { namespace bg {

class HSVCylinderFeatureExtractor : public PixelFeatureExtractor
{
public:
    HSVCylinderFeatureExtractor(double scaleOfV)
        : scaleOfV(scaleOfV){}

    virtual auto getFeatures(cv::Mat const& img) const -> cv::Mat1d;

    CVX_CLONE_IN_DERIVED(HSVCylinderFeatureExtractor)

private:
    double scaleOfV;
};

}}

#endif // HSVCYLINDERFEATUREEXTRACTOR_HPP
