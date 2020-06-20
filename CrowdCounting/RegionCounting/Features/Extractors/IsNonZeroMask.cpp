#include <cvextra/core.hpp>
#include <cvextra/mats.hpp>
#include <CrowdCounting/RegionCounting/Features/Extractors/IsNonZeroMask.hpp>
#include <CrowdCounting/RegionCounting/Features/PreprocessedFrame.hpp>
#include <opencv2/core/core.hpp>
#include <string>
#include <vector>

using namespace std;
using namespace cv;
using namespace cvx;
using namespace crowd;

auto IsNonZeroMask::
extract(
		PreprocessedFrame const& frame,
		Rectd relativeRect
		) const -> std::vector<double>
{
    Mat maskPart = cvx::extractRelativeRoi(frame.mask, relativeRect);
    return {cv::countNonZero(maskPart) == 0 ? 0.0 : 1.0};
}

int IsNonZeroMask::
getFeatureCount() const
{
    return 1;
}

auto IsNonZeroMask::
getNames() const -> std::vector<string>
{
    return {"is nonzero mask"};
}

auto IsNonZeroMask::
getDescription() const -> string
{
    return "Is nonzero mask";
}

auto IsNonZeroMask::
describe() const -> boost::property_tree::ptree
{
	boost::property_tree::ptree pt;
	pt.put("type", "IsNonZeroMask");
	return pt;
}

auto IsNonZeroMask::
create(boost::property_tree::ptree const& pt) -> std::unique_ptr<IsNonZeroMask>
{
	return stdx::make_unique<IsNonZeroMask>();
}
