#include <boost/algorithm/string/join.hpp>
#include <cvextra/vectors.hpp>
#include <CrowdCounting/RegionCounting/Features/MultiFeatureExtractor.hpp>
#include <stdx/stdx.hpp>
#include <initializer_list>
#include <memory>
#include <string>

using namespace std;
using namespace cv;
using namespace cvx;
using namespace crowd;

MultiFeatureExtractor::
MultiFeatureExtractor(
        std::vector<stdx::any_reference_wrapper<FeatureExtractor const>> _extractors)
{
    for (FeatureExtractor const& extr : _extractors)
    {
        extractors.push_back(extr.clone());
    }
}

MultiFeatureExtractor::
MultiFeatureExtractor(
        std::initializer_list<stdx::any_reference_wrapper<FeatureExtractor const>> _extractors)
{
    for (FeatureExtractor const& extr : _extractors)
    {
        extractors.push_back(extr.clone());
    }
}

auto MultiFeatureExtractor::
extractFeatures(
		PreprocessedFrame const& frame,
		Rectd relativeRect
		) const -> vector<double>
{
    vector<double> result;
    for (auto& extr : extractors)
    {
        cvx::vectors::push_back_all(result, extr->extract(frame, relativeRect));
    }
    return result;
}

auto MultiFeatureExtractor::
getFeatureCount() const -> int
{
    int result = 0;
    for (auto& extr : extractors)
    {
        result += extr->getFeatureCount();
    }
    return result;
}

auto MultiFeatureExtractor::
getNames() const -> vector<string>
{
    vector<string> result;
    for (auto& extr : extractors)
    {
        cvx::vectors::push_back_all(result, extr->getNames());
    }
    return result;
}

auto MultiFeatureExtractor::
getDescription() const -> string
{
    vector<string> descriptions;
    for (auto& extr : extractors)
    {
        descriptions.push_back(extr->getDescription());
    }
    return boost::algorithm::join(descriptions, "\n");
}

auto MultiFeatureExtractor::
describe() const -> boost::property_tree::ptree
{
	boost::property_tree::ptree pt;

	for (auto const& ex : extractors)
	{
		pt.add_child("feature", ex->describe());
	}

	return pt;
}

auto MultiFeatureExtractor::
create(boost::property_tree::ptree const& pt) -> std::unique_ptr<MultiFeatureExtractor>
{
	vector<unique_ptr<FeatureExtractor>> extractors;
	vector<stdx::any_reference_wrapper<FeatureExtractor const>> extractorRefereces;

	for (auto const& elem : pt)
	{
		extractors.emplace_back(std::move(FeatureExtractor::create(elem.second)));
		extractorRefereces.emplace_back(*extractors.back());
	}

	return stdx::make_unique<MultiFeatureExtractor>(extractorRefereces);
}

auto crowd::MultiFeatureExtractor::
preprocess(const CountingFrame& cf) const -> PreprocessedFrame
{
	return PreprocessedFrame{cf};
}
