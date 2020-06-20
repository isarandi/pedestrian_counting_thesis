#ifndef PIXELFEATUREEXTRACTOR_HPP
#define PIXELFEATUREEXTRACTOR_HPP

#include <opencv2/core/core.hpp>
#include <stdx/cloning.hpp>
#include <memory>

namespace crowd { namespace bg {

class PixelFeatureExtractor
{
public:
    virtual auto getFeatures(cv::Mat const& img) const -> cv::Mat1d = 0;
    virtual ~PixelFeatureExtractor(){}

    CVX_CLONE_IN_BASE(PixelFeatureExtractor)
};

}}

#endif // PIXELFEATUREEXTRACTOR_HPP
