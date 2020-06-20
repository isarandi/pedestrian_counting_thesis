#include <CrowdCounting/LineCounting/Features/Extractors/Foreground.hpp>
#include <cvextra/core.hpp>
#include <cvextra/cvret.hpp>
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

auto crowd::linecounting::Foreground::
extract(
		FeatureSlices const& slices,
		BinaryMat segmentMask
		) const -> cv::Mat1d
{
    BinaryMat movingForegroundMask = slices.foregroundMask.clone();
    movingForegroundMask.setTo(0.0, segmentMask==0);
    movingForegroundMask.setTo(0.0, slices.stencil==0);

    double movingForegroundSize = cv::mean(movingForegroundMask)[0];
	return cvx::m({{movingForegroundSize}});
}

auto crowd::linecounting::Foreground::
describe() const -> boost::property_tree::ptree
{
	boost::property_tree::ptree pt;
	pt.put("type", "Foreground");
	return pt;
}

auto crowd::linecounting::Foreground::
create(boost::property_tree::ptree const& pt) -> std::unique_ptr<Foreground>
{
	return stdx::make_unique<Foreground>();
}
