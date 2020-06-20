#ifndef CROWDCOUNTING_LineCounting_FEATURES_EXTRACTORS_DUMMY_HPP_
#define CROWDCOUNTING_LineCounting_FEATURES_EXTRACTORS_DUMMY_HPP_

#include "../LineFeatureExtractor.hpp"

namespace crowd {
namespace linecounting {

class Dummy : public LineFeatureExtractor
{
public:

    virtual auto extract(
            FeatureSlices const& slices,
			cvx::BinaryMat segmentMask
			) const -> cv::Mat1d;

    CVX_CLONE_IN_DERIVED(Dummy)
    CVX_CONFIG_DERIVED(Dummy)

};

} /* namespace linecounting */
} /* namespace crowd */

#endif /* CROWDCOUNTING_LineCounting_FEATURES_EXTRACTORS_DUMMY_HPP_ */
