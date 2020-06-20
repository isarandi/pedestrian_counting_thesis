#include <cvextra/core.hpp>
#include <cvextra/improc.hpp>
#include <cvextra/mats.hpp>
#include <CrowdCounting/RegionCounting/Features/Extractors/Perimeter.hpp>
#include <CrowdCounting/RegionCounting/Features/Impl/features.hpp>
#include <CrowdCounting/RegionCounting/Features/PreprocessedFrame.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/core/mat.hpp>
#include <string>
#include <vector>

using namespace std;
using namespace cv;
using namespace cvx;
using namespace crowd;

auto Perimeter::
extract(
		PreprocessedFrame const& frame,
		Rectd relativeRect
		) const -> std::vector<double>
{
    Mat maskPart = cvx::extractRelativeRoi(frame.mask, relativeRect);
    Mat outline = cvxret::outline(maskPart);
    Mat scalePart = cvx::extractRelativeRoi(frame.scaleMap, relativeRect);

    return {crowd::weightedPixelCount(outline, 1.0/scalePart)};
}

auto Perimeter::
getFeatureCount() const -> int
{
    return 1;
}

auto Perimeter::
getNames() const -> std::vector<string>
{
    return {"foreground perimeter length"};
}

auto Perimeter::
getDescription() const -> string
{
    return "Foreground perimeter length";
}

auto Perimeter::
describe() const -> boost::property_tree::ptree
{
	boost::property_tree::ptree pt;
	pt.put("type", "Perimeter");
	return pt;
}

auto Perimeter::
create(boost::property_tree::ptree const& pt) -> std::unique_ptr<Perimeter>
{
	return stdx::make_unique<Perimeter>();
}
