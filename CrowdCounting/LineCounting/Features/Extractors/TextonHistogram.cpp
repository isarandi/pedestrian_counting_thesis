#include <CrowdCounting/LineCounting/Features/Extractors/TextonHistogram.hpp>
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

auto crowd::linecounting::TextonHistogram::
extract(
		FeatureSlices const& slices,
		BinaryMat segmentMask
		) const -> cv::Mat1d
{
    Mat1b maskedTexton = slices.textonMap.clone()+1;
    maskedTexton.setTo(0, segmentMask==0);
    maskedTexton.setTo(0, slices.stencil==0);

    Mat1d textonHist{1, slices.nTextons};
    for (int iTexton : cvx::irange(slices.nTextons))
    {
    	textonHist(iTexton) = cv::mean(slices.textonMap==iTexton)[0];
    }

    return textonHist;
}

auto crowd::linecounting::TextonHistogram::
describe() const -> boost::property_tree::ptree
{
	boost::property_tree::ptree pt;
	pt.put("type", "TextonHistogram");
	return pt;
}

auto crowd::linecounting::TextonHistogram::
create(boost::property_tree::ptree const& pt) -> std::unique_ptr<TextonHistogram>
{
	return stdx::make_unique<TextonHistogram>();
}
