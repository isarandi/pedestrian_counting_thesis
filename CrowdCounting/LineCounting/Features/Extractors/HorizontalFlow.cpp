#include <CrowdCounting/LineCounting/Features/Extractors/HorizontalFlow.hpp>
#include <cvextra/core.hpp>
#include <cvextra/cvret.hpp>
#include <cvextra/LoopRange.hpp>
#include <cvextra/visualize.hpp>
#include <CrowdCounting/LineCounting/FeatureSlices.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/core/mat.hpp>
#include <cmath>
#include <string>
#include <vector>

using namespace std;
using namespace cv;
using namespace cvx;
using namespace crowd;

auto crowd::linecounting::HorizontalFlow::
extract(
		FeatureSlices const& slices,
		BinaryMat segmentMask
		) const -> cv::Mat1d
{
    Mat1d maskedXFlow = cvret::extractChannel(slices.flow, 0);
    maskedXFlow.setTo(0.0, segmentMask==0);
    maskedXFlow.setTo(0.0, slices.stencil==0);

    double horizFlowInMask = std::abs(cv::mean(maskedXFlow)[0]);
    return cvx::m({{horizFlowInMask}});
}

auto crowd::linecounting::HorizontalFlow::
describe() const -> boost::property_tree::ptree
{
	boost::property_tree::ptree pt;
	pt.put("type", "HorizontalFlow");
	return pt;
}

auto crowd::linecounting::HorizontalFlow::
create(boost::property_tree::ptree const& pt) -> std::unique_ptr<HorizontalFlow>
{
	return stdx::make_unique<HorizontalFlow>();
}
