#ifndef CROWDCOUNTING_LineCounting_SEGMENTDESCRIBER_HPP_
#define CROWDCOUNTING_LineCounting_SEGMENTDESCRIBER_HPP_

#include <cvextra/core.hpp>
#include <CrowdCounting/RegionCounting/Features/PreprocessedFrame.hpp>
#include <CrowdCounting/LineCounting/FeatureSlices.hpp>
#include <cvextra/configfile.hpp>
#include <stdx/cloning.hpp>
#include <memory>
#include <string>
#include <vector>

namespace crowd
{

namespace linecounting {

class LineFeatureExtractor
{
public:
    virtual auto extract(
            FeatureSlices const& slices,
			cvx::BinaryMat segmentMask
			) const -> cv::Mat1d = 0;

    virtual ~LineFeatureExtractor(){}
    CVX_CLONE_IN_BASE(LineFeatureExtractor)
    CVX_CONFIG_BASE(LineFeatureExtractor)
};


}

} /* namespace crowd */

#endif /* CROWDCOUNTING_LineCounting_SEGMENTDESCRIBER_HPP_ */
