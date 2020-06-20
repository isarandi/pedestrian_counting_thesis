#include <CrowdCounting/LineCounting/Features/Extractors/Dummy.hpp>
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

auto crowd::linecounting::Dummy::
extract(
		FeatureSlices const& slices,
		BinaryMat segmentMask
		) const -> cv::Mat1d
{
	return cvx::m({{0}});
}

auto crowd::linecounting::Dummy::
describe() const -> boost::property_tree::ptree
{
	boost::property_tree::ptree pt;
	pt.put("type", "Dummy");
	return pt;
}

auto crowd::linecounting::Dummy::
create(boost::property_tree::ptree const& pt) -> std::unique_ptr<Dummy>
{
	return stdx::make_unique<Dummy>();
}
