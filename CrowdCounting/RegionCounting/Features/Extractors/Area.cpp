#include <cvextra/core.hpp>
#include <cvextra/mats.hpp>
#include <CrowdCounting/RegionCounting/Features/Extractors/Area.hpp>
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

std::vector<double> Area::extract(PreprocessedFrame const& frame, Rectd relativeRect) const
{
    Mat maskPart = cvx::extractRelativeRoi(frame.mask, relativeRect);
    Mat scalePart = cvx::extractRelativeRoi(frame.scaleMap, relativeRect);
    return {crowd::weightedPixelCount(maskPart, 1.0/scalePart.mul(scalePart))};
}

int Area::getFeatureCount() const
{
    return 1;
}

std::vector<string> Area::getNames() const
{
    return {"foreground area"};
}

string Area::getDescription() const
{
    return "Foreground area";
}

auto Area::
describe() const -> boost::property_tree::ptree
{
	boost::property_tree::ptree pt;
	pt.put("type", "Area");
	return pt;
}

auto Area::
create(boost::property_tree::ptree const& pt) -> std::unique_ptr<Area>
{
	return stdx::make_unique<Area>();
}
