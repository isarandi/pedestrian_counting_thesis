#include <cvextra/core.hpp>
#include <cvextra/math.hpp>
#include <cvextra/mats.hpp>
#include <cvextra/strings.hpp>
#include <CrowdCounting/RegionCounting/Features/Extractors/StatisticalLandscape.hpp>
#include <CrowdCounting/RegionCounting/Features/Impl/features.hpp>
#include <CrowdCounting/RegionCounting/Features/PreprocessedFrame.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/core/mat.hpp>
#include <stdx/stdx.hpp>
#include <sstream>
#include <string>
#include <vector>

using namespace std;
using namespace cv;
using namespace cvx;
using namespace crowd;

StatisticalLandscape::
StatisticalLandscape(int nThresholds, bool _masked)
    : masked(_masked)
{
    thresholds = cvx::math::linspace2(255.0/(nThresholds+1.0), 255.0, nThresholds);
}

StatisticalLandscape::
StatisticalLandscape(std::vector<double> const& thresholds, bool _masked)
    : masked(_masked)
	, thresholds(thresholds)
{
}

auto StatisticalLandscape::
extract(
		PreprocessedFrame const& frame,
		Rectd relativeRect
		) const -> std::vector<double>
{
    Mat grayPart = cvx::extractRelativeRoi(frame.grayFrame, relativeRect);
    Mat maskPart = cvx::extractRelativeRoi(frame.mask, relativeRect);

    Mat inputToStatLandscape = (masked ? cv::min(grayPart, maskPart) : grayPart);

    return crowd::statisticalLandscape(inputToStatLandscape, thresholds);
}

auto StatisticalLandscape::
getFeatureCount() const -> int
{
    return thresholds.size()*4;
}

auto StatisticalLandscape::
getNames() const -> std::vector<string>
{
    vector<string> names;
    for (auto& threshold : thresholds)
    {
        string threshStr = std::to_string(threshold);
        names.push_back("statistical landscape thresh=" + threshStr + " nUpperComp");
        names.push_back("statistical landscape thresh=" + threshStr + " meanDiffUpperComp");
        names.push_back("statistical landscape thresh=" + threshStr + " nLowerComp");
        names.push_back("statistical landscape thresh=" + threshStr + " meanDiffLowerComp");
    }
    return names;
}

auto StatisticalLandscape::
getDescription() const -> string
{
    stringstream ss;
    ss << "Thresholds: " << thresholds;
    return "Statistical Landscape\n" + cvx::str::indentBlock(ss.str());
}

auto StatisticalLandscape::
describe() const -> boost::property_tree::ptree
{
	boost::property_tree::ptree pt;
	pt.put("type", "StatisticalLandscape");
	pt.put("masked", masked);

	boost::property_tree::ptree ptThresholds;
	for (auto const& elem : thresholds)
	{
		ptThresholds.add("threshold", elem);
	}
	pt.add_child("thresholds", ptThresholds);

	return pt;
}

auto StatisticalLandscape::
create(boost::property_tree::ptree const& pt) -> std::unique_ptr<StatisticalLandscape>
{
	bool masked = pt.get<bool>("masked");
	vector<double> thresholds;

	auto bounds = pt.get_child("thresholds").equal_range("threshold");

	for (auto it = bounds.first; it != bounds.second ; ++it)
	{
	    thresholds.push_back((*it).second.get_value<int>());
	}

	return stdx::make_unique<StatisticalLandscape>(thresholds, masked);
}
