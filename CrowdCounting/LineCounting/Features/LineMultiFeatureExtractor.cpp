#include <CrowdCounting/LineCounting/Features/LineMultiFeatureExtractor.hpp>
#include <boost/algorithm/string/join.hpp>
#include <cvextra/vectors.hpp>
#include <stdx/stdx.hpp>
#include <initializer_list>
#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/ptree_fwd.hpp>
#include <memory>
#include <string>

using namespace std;
using namespace cv;
using namespace cvx;
using namespace crowd;
using namespace crowd::linecounting;

LineMultiFeatureExtractor::
LineMultiFeatureExtractor(
        std::vector<stdx::any_reference_wrapper<LineFeatureExtractor const>> _extractors)
{
    for (LineFeatureExtractor const& extr : _extractors)
    {
        extractors.push_back(extr.clone());
    }
}

LineMultiFeatureExtractor::
LineMultiFeatureExtractor(
        std::initializer_list<stdx::any_reference_wrapper<LineFeatureExtractor const>> _extractors)
{
    for (LineFeatureExtractor const& extr : _extractors)
    {
        extractors.push_back(extr.clone());
    }
}

auto LineMultiFeatureExtractor::
extractFeatures(
        FeatureSlices const& slices,
		BinaryMat segmentMask
		) const -> Mat1d
{
    vector<double> result;
    for (auto& extr : extractors)
    {
        cvx::vectors::push_back_all(result, extr->extract(slices, segmentMask));
    }
    return Mat1d(result,true).t();
}

auto LineMultiFeatureExtractor::
describe() const -> boost::property_tree::ptree
{
	boost::property_tree::ptree pt;

	for (auto const& ex : extractors)
	{
		pt.add_child("feature", ex->describe());
	}

	return pt;
}

auto LineMultiFeatureExtractor::
create(boost::property_tree::ptree const& pt) -> std::unique_ptr<LineMultiFeatureExtractor>
{
	vector<unique_ptr<LineFeatureExtractor>> extractors;
	vector<stdx::any_reference_wrapper<LineFeatureExtractor const>> extractorRefereces;

	for (auto const& elem : pt)
	{
		extractors.emplace_back(std::move(LineFeatureExtractor::create(elem.second)));
		extractorRefereces.emplace_back(*extractors.back());
	}

	return stdx::make_unique<LineMultiFeatureExtractor>(extractorRefereces);
}

