#include <CrowdCounting/RegionCounting/Features/Extractors/RegionTextonHistogram.hpp>
#include <cvextra/core.hpp>
#include <cvextra/mats.hpp>
#include <cvextra/LoopRange.hpp>
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

auto RegionTextonHistogram::
extract(
		PreprocessedFrame const& frame,
		Rectd relativeRect
		) const -> std::vector<double>
{
    Mat1b maskedTexton = frame.textonMap.clone();

    int nTextons = 6;
    vector<double> textonHist(nTextons);
    for (int iTexton : cvx::irange(nTextons))
    {
    	textonHist[iTexton] = cv::mean(frame.textonMap==(iTexton+1))[0];
    }

    return textonHist;
}

auto RegionTextonHistogram::
getFeatureCount() const -> int
{
    return 6;
}

auto RegionTextonHistogram::
getNames() const -> std::vector<string>
{
    return {"textonhist"};
}

auto RegionTextonHistogram::
getDescription() const -> string
{
    return "Texton Histogram";
}


auto RegionTextonHistogram::
describe() const -> boost::property_tree::ptree
{
	boost::property_tree::ptree pt;
	pt.put("type", "RegionTextonHistogram");
	return pt;
}

auto RegionTextonHistogram::
create(boost::property_tree::ptree const& pt) -> std::unique_ptr<RegionTextonHistogram>
{
	return stdx::make_unique<RegionTextonHistogram>();
}
