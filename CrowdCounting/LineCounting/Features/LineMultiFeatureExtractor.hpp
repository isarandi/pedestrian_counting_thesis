#ifndef CROWDCOUNTING_LineCounting_FEATURES_LINEMULTIFEATUREEXTRACTOR_HPP_
#define CROWDCOUNTING_LineCounting_FEATURES_LINEMULTIFEATUREEXTRACTOR_HPP_

#include <CrowdCounting/LineCounting/Features/LineFeatureExtractor.hpp>
#include <cvextra/core.hpp>
#include <cvextra/configfile.hpp>
#include <stdx/stdx.hpp>
#include <initializer_list>
#include <memory>
#include <string>
#include <vector>

namespace crowd {
class FeatureSlices;
} /* namespace crowd */



namespace crowd {
namespace linecounting {

class LineMultiFeatureExtractor
{
public:
	LineMultiFeatureExtractor(
            std::vector<stdx::any_reference_wrapper<LineFeatureExtractor const>> extractors);
	LineMultiFeatureExtractor(
            std::initializer_list<stdx::any_reference_wrapper<LineFeatureExtractor const>> extractors);

    auto extractFeatures(
            FeatureSlices const& slices,
			cvx::BinaryMat segmentMask
			) const -> cv::Mat1d;

    //virtual auto describe() const -> boost::property_tree::ptree = 0;

    CVX_CONFIG_SINGLE(LineMultiFeatureExtractor)

private:

    std::vector<std::shared_ptr<LineFeatureExtractor const>> extractors;
};

} /* namespace linecounting */
} /* namespace crowd */

#endif /* CROWDCOUNTING_LineCounting_FEATURES_LINEMULTIFEATUREEXTRACTOR_HPP_ */
