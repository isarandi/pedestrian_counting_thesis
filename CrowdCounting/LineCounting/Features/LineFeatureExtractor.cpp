#include <CrowdCounting/LineCounting/Features/LineFeatureExtractor.hpp>
#include <CrowdCounting/LineCounting/Features/Extractors/HorizontalFlow.hpp>
#include <CrowdCounting/LineCounting/Features/Extractors/CannySlice.hpp>
#include <CrowdCounting/LineCounting/Features/Extractors/TextonHistogram.hpp>
#include <CrowdCounting/LineCounting/Features/Extractors/Foreground.hpp>

using namespace crowd::linecounting;

auto LineFeatureExtractor::
create(
		boost::property_tree::ptree const& pt
		) -> std::unique_ptr<LineFeatureExtractor>
{
	std::string type = pt.get<std::string>("type");
	if (type == "HorizontalFlow")
	{
		return HorizontalFlow::create(pt);
	} else if (type == "CannySlice")
	{
		return CannySlice::create(pt);
	} else if (type == "TextonHistogram")
	{
		return TextonHistogram::create(pt);
	} else if (type == "Foreground")
	{
		return Foreground::create(pt);
	}
	throw 1;
}

