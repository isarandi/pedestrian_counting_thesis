#include <CrowdCounting/RegionCounting/RegionCounter.hpp>
#include <CrowdCounting/RegionCounting/AllToAllRegionCounter.hpp>
#include <CrowdCounting/RegionCounting/SharedModelRegionCounter.hpp>

using namespace crowd;

auto RegionCounter::
create(
		boost::property_tree::ptree const& pt
		) -> std::unique_ptr<RegionCounter>
{
	std::string type = pt.get<std::string>("type");
	if (type == "AllToAllRegionCounter")
	{
		return AllToAllRegionCounter::create(pt);
	} else if (type == "SharedModelRegionCounter")
	{
		return SharedModelRegionCounter::create(pt);
	}
	throw 1;
}

