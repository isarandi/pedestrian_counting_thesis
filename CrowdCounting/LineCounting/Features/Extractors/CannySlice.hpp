#ifndef CROWDCOUNTING_LineCounting_FEATURES_EXTRACTORS_CANNNYSLICE_HPP_
#define CROWDCOUNTING_LineCounting_FEATURES_EXTRACTORS_CANNNYSLICE_HPP_

#include "../LineFeatureExtractor.hpp"

namespace crowd {
namespace linecounting {

class CannySlice : public LineFeatureExtractor
{
public:

    virtual auto extract(
            FeatureSlices const& slices,
			cvx::BinaryMat segmentMask
			) const -> cv::Mat1d;

    CVX_CLONE_IN_DERIVED(CannySlice)
    CVX_CONFIG_DERIVED(CannySlice)

};

} /* namespace linecounting */
} /* namespace crowd */

#endif /* CROWDCOUNTING_LineCounting_FEATURES_EXTRACTORS_HORIZONTALFLOW_HPP_ */
