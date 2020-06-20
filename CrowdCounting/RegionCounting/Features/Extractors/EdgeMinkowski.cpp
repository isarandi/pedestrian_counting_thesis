#include <cvextra/core.hpp>
#include <cvextra/mats.hpp>
#include <cvextra/strings.hpp>
#include <CrowdCounting/RegionCounting/Features/Extractors/EdgeMinkowski.hpp>
#include <CrowdCounting/RegionCounting/Features/Impl/features.hpp>
#include <CrowdCounting/RegionCounting/Features/PreprocessedFrame.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/core/mat.hpp>
#include <sstream>
#include <string>
#include <vector>

using namespace std;
using namespace cv;
using namespace cvx;
using namespace crowd;

auto EdgeMinkowski::
extract(
		PreprocessedFrame const& frame,
		Rectd relativeRect
		) const -> vector<double>
{
    BinaryMat edgesPart = cvx::extractRelativeRoi(frame.edges, relativeRect);

    if (masked)
    {
        BinaryMat maskPart = cvx::extractRelativeRoi(frame.mask, relativeRect);
        edgesPart = cv::min(edgesPart, maskPart);
    }

    return {crowd::Minkowski(edgesPart, maxRadius)};
}

int EdgeMinkowski::
getFeatureCount() const
{
    return 1;
}

auto EdgeMinkowski::
getNames() const -> vector<string>
{
    return {"edge minkowski"};
}

auto EdgeMinkowski::
getDescription() const -> string
{
    stringstream ss;

    ss << "Max radius: " << maxRadius << endl;
    ss << "Masked: " << masked;

    return "Edge Minkowski\n" + cvx::str::indentBlock(ss.str());
}

auto EdgeMinkowski::
describe() const -> boost::property_tree::ptree
{
	boost::property_tree::ptree pt;
	pt.put("type", "EdgeMinkowski");
	pt.put("masked", masked);
	pt.put("maxRadius", maxRadius);
	return pt;
}

auto EdgeMinkowski::
create(boost::property_tree::ptree const& pt) -> std::unique_ptr<EdgeMinkowski>
{
	return stdx::make_unique<EdgeMinkowski>(pt.get<int>("maxRadius"), pt.get<bool>("masked"));
}


