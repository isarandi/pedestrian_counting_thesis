#include <cvextra/coords.hpp>
#include <cvextra/mats.hpp>
#include <cvextra/strings.hpp>
#include <CrowdCounting/RegionCounting/Features/Extractors/LocalBinaryPatterns.hpp>
#include <CrowdCounting/RegionCounting/Features/Impl/features.hpp>
#include <CrowdCounting/RegionCounting/Features/Impl/localBinaryPatternsImpl.hpp>
#include <opencv2/core/mat.hpp>
#include <opencv2/core/core.hpp>
#include <sstream>

using namespace std;
using namespace cv;
using namespace cvx;
using namespace crowd;

auto LocalBinaryPatterns::
extract(
		PreprocessedFrame const& frame,
		Rectd relativeRect
		) const -> std::vector<double>
{
    Mat1b grayPart = cvx::extractRelativeRoi(frame.grayFrame, relativeRect);

    double pxPerMeter = frame.scaleMap(cvx::center(frame.scaleMap));

    //Mat1b blurred = cvret::GaussianBlur(grayPart, Size(11,11), pxPerMeter*0.02);

    return crowd::localBinaryPatternHistograms(grayPart, radius, nSamples, nHistogramBins, subGrid);
}

auto LocalBinaryPatterns::
getFeatureCount() const -> int
{
    return subGrid.area() * nHistogramBins;
}

auto LocalBinaryPatterns::
getNames() const -> std::vector<string>
{
    return crowd::createIndexSuffixedVersions("local binary patterns ", getFeatureCount());
}

auto LocalBinaryPatterns::
getDescription() const -> string
{
    stringstream ss;

    ss << "Radius: " << radius << endl;
    ss << "Sample count: " << nSamples << endl;
    ss << "Histogram bins: " << nHistogramBins << endl;
    ss << "Subgrid size: " << subGrid.width << "x" << subGrid.height << endl;
    ss << "Masked: " << masked;

    return "Local Binary Patterns\n" + cvx::str::indentBlock(ss.str());
}

auto LocalBinaryPatterns::
describe() const -> boost::property_tree::ptree
{
	boost::property_tree::ptree pt;
	pt.put("type", "LocalBinaryPatterns");
	pt.put("radius", radius);
	pt.put("nSamples", nHistogramBins);
	pt.put("nHistogramBins", nHistogramBins);
	pt.put("subGrid.width", subGrid.width);
	pt.put("subGrid.height", subGrid.height);
	pt.put("masked", masked);
	return pt;
}

auto LocalBinaryPatterns::
create(boost::property_tree::ptree const& pt) -> std::unique_ptr<LocalBinaryPatterns>
{
	return stdx::make_unique<LocalBinaryPatterns>(
			pt.get<double>("radius"),
			pt.get<int>("nSamples"),
			pt.get<int>("nHistogramBins"),
			cv::Size{pt.get<int>("subGrid.width"), pt.get<int>("subGrid.height")},
			pt.get<bool>("masked")
	);
}
