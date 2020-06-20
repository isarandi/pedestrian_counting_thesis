#include <CrowdCounting/RegionCounting/Features/FeatureExtractor.hpp>
#include <CrowdCounting/RegionCounting/Features/FeatureExtractors.hpp>

using namespace crowd;

auto FeatureExtractor::
create(
		boost::property_tree::ptree const& pt
) -> std::unique_ptr<FeatureExtractor>
{
	std::string type = pt.get<std::string>("type");

	if (type == "Area")
	{
		return Area::create(pt);
	}
	else if (type == "Perimeter")
	{
		return Perimeter::create(pt);
	}
	else if (type == "PerimeterToAreaRatio")
	{
		return PerimeterToAreaRatio::create(pt);
	}
	else if (type == "EdgeOriHistogram")
	{
		return EdgeOriHistogram::create(pt);
	}
	else if (type == "EdgeMinkowski")
	{
		return EdgeMinkowski::create(pt);
	}
	else if (type == "LocalBinaryPatterns")
	{
		return LocalBinaryPatterns::create(pt);
	}
	else if (type == "StatisticalLandscape")
	{
		return StatisticalLandscape::create(pt);
	}
	else if (type == "IsNonZeroMask")
	{
		return IsNonZeroMask::create(pt);
	}
	else if (type == "RegionTextonHistogram")
	{
		return RegionTextonHistogram::create(pt);
	}
	else if (type == "PerimeterOriHistogram")
	{
		return PerimeterOriHistogram::create(pt);
	}
	else if (type == "GrayLevelCooccurrence")
	{
		return GrayLevelCooccurrence::create(pt);
	}
	else if (type == "FilterBank")
	{
		return FilterBank::create(pt);
	}
	throw 1;
}
