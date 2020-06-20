#include <cvextra/core.hpp>
#include <cvextra/mats.hpp>
#include <cvextra/vectors.hpp>
#include <CrowdCounting/RegionCounting/Features/Extractors/GrayLevelCooccurrence.hpp>
#include <CrowdCounting/RegionCounting/Features/Impl/features.hpp>
#include <CrowdCounting/RegionCounting/Features/PreprocessedFrame.hpp>
#include <opencv2/core/core.hpp>
#include <string>
#include <vector>

using namespace std;
using namespace cv;
using namespace cvx;
using namespace crowd;

auto GrayLevelCooccurrence::
extract(
        PreprocessedFrame const& frame,
        Rectd relativeRect
		) const -> std::vector<double>
{
    Mat maskPart = cvx::extractRelativeRoi(frame.mask, relativeRect);
    Mat grayPart = cvx::extractRelativeRoi(frame.grayFrame, relativeRect);

    return crowd::GLCMFeatures(grayPart, maskPart);
}

auto GrayLevelCooccurrence::
getFeatureCount() const -> int
{
    return 12;
}

auto GrayLevelCooccurrence::
getNames() const -> std::vector<string>
{
    vector<string> result;
    cvx::vectors::push_back_all(result, crowd::createIndexSuffixedVersions("GLCM homogeneity",4));
    cvx::vectors::push_back_all(result, crowd::createIndexSuffixedVersions("GLCM entropy",4));
    cvx::vectors::push_back_all(result, crowd::createIndexSuffixedVersions("GLCM energy",4));

    return result;
}

auto GrayLevelCooccurrence::
getDescription() const -> string
{
    return "Gray level co-occurrence features";
}

auto GrayLevelCooccurrence::
describe() const -> boost::property_tree::ptree
{
	boost::property_tree::ptree pt;
	pt.put("type", "GrayLevelCooccurrence");
	return pt;
}

auto GrayLevelCooccurrence::
create(boost::property_tree::ptree const& pt) -> std::unique_ptr<GrayLevelCooccurrence>
{
	return stdx::make_unique<GrayLevelCooccurrence>();
}
