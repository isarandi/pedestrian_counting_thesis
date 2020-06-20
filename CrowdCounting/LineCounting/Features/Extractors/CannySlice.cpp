#include <CrowdCounting/LineCounting/Features/Extractors/CannySlice.hpp>
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

auto crowd::linecounting::CannySlice::
extract(
		FeatureSlices const& slices,
		BinaryMat segmentMask
		) const -> cv::Mat1d
{
	Mat1d maskedCanny = slices.canny.clone();
	maskedCanny.setTo(0.0, segmentMask==0);
	maskedCanny.setTo(0.0, slices.stencil==0);

	double cannySize = std::abs(cv::mean(maskedCanny/255.0)[0]);
	return cvx::mats::matFromRows({{cannySize}});
}

auto crowd::linecounting::CannySlice::
describe() const -> boost::property_tree::ptree
{
	boost::property_tree::ptree pt;
	pt.put("type", "CannySlice");
	return pt;
}

auto crowd::linecounting::CannySlice::
create(boost::property_tree::ptree const& pt) -> std::unique_ptr<CannySlice>
{
	return stdx::make_unique<CannySlice>();
}
