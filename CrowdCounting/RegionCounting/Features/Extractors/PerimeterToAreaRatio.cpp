#include <cvextra/core.hpp>
#include <CrowdCounting/RegionCounting/Features/Extractors/Area.hpp>
#include <CrowdCounting/RegionCounting/Features/Extractors/Perimeter.hpp>
#include <CrowdCounting/RegionCounting/Features/Extractors/PerimeterToAreaRatio.hpp>
#include <CrowdCounting/RegionCounting/Features/PreprocessedFrame.hpp>
#include <string>
#include <vector>

using namespace std;
using namespace cv;
using namespace cvx;
using namespace crowd;

auto PerimeterToAreaRatio::
extract(
		PreprocessedFrame const& frame,
		Rectd relativeRect
		) const -> vector<double>
{
    double area = Area{}.extract(frame, relativeRect)[0];

    if (area == 0)
        return {0};

    double perimeter = Perimeter{}.extract(frame, relativeRect)[0];
    return {perimeter/area};
}

int PerimeterToAreaRatio::
getFeatureCount() const
{
    return 1;
}

auto PerimeterToAreaRatio::
getNames() const -> std::vector<string>
{
    return {"perimeter to area ratio"};
}

auto PerimeterToAreaRatio::
getDescription() const -> string
{
    return "Perimeter to area ratio";
}

auto PerimeterToAreaRatio::
describe() const -> boost::property_tree::ptree
{
	boost::property_tree::ptree pt;
	pt.put("type", "PerimeterToAreaRatio");
	return pt;
}

auto PerimeterToAreaRatio::
create(boost::property_tree::ptree const& pt) -> std::unique_ptr<PerimeterToAreaRatio>
{
	return stdx::make_unique<PerimeterToAreaRatio>();
}
