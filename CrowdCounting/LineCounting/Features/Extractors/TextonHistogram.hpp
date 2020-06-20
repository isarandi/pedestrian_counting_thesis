#ifndef CROWDCOUNTING_LINECOUNTING_FEATURES_EXTRACTORS_TEXTONHISTOGRAM_HPP_
#define CROWDCOUNTING_LINECOUNTING_FEATURES_EXTRACTORS_TEXTONHISTOGRAM_HPP_

#include "../LineFeatureExtractor.hpp"

namespace crowd {
namespace linecounting {

class TextonHistogram : public LineFeatureExtractor
{
public:
    virtual auto extract(
            FeatureSlices const& slices,
			cvx::BinaryMat segmentMask
			) const -> cv::Mat1d;

    CVX_CLONE_IN_DERIVED(TextonHistogram)
    CVX_CONFIG_DERIVED(TextonHistogram)

};

} /* namespace linecounting */
} /* namespace crowd */

#endif /* CROWDCOUNTING_LINECOUNTING_FEATURES_EXTRACTORS_TEXTONHISTOGRAM_HPP_ */
