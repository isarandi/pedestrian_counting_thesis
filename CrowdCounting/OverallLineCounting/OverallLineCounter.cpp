#include <CrowdCounting/OverallLineCounting/OverallLineCounter.hpp>
#include <CrowdCounting/OverallLineCounting/FullResult.hpp>
#include <CrowdCounting/OverallLineCounting/SharedModelOverallLineCounter.hpp>
#include <CrowdCounting/OverallLineCounting/LineAndRegionCombiner.hpp>
#include <CrowdCounting/LineCounting/FlowMosaicking/FlowMosaicCounter.hpp>
#include <Run/config.hpp>

using namespace std;
using namespace cv;
using namespace cvx;
using namespace crowd::linecounting;

auto OverallLineCountingSet::
makeMany(
		std::vector<std::string> const& names,
		cv::Range const& frameRange) -> vector<OverallLineCountingSet>
{
	vector<OverallLineCountingSet> result;
	for (auto const& name : names)
	{
		result.push_back(OverallLineCountingSet{name, frameRange});
	}
	return result;
}

auto OverallLineCounter::
create(
		boost::property_tree::ptree const& pt
		) -> std::unique_ptr<OverallLineCounter>
{
	std::string type = pt.get<std::string>("type");
	if (type == "SharedModelOverallLineCounter")
	{
		return SharedModelOverallLineCounter::create(pt);
	} else if (type == "FlowMosaicCounter")
	{
	    return FlowMosaicCounter::create(pt);
	}
	throw 1;
}

auto OverallLineCountingSet::
getLineCountingSet(cvx::LineSegment const& seg) const -> LineCountingSet
{
    return LineCountingSet{datasetName, seg, frameRange};
}

auto OverallLineCountingSet::
describe() const -> boost::property_tree::ptree
{
	boost::property_tree::ptree pt;
	pt.put("datasetName", datasetName);
	pt.put("frameRange.start", frameRange.start);
	pt.put("frameRange.end", frameRange.end);
	return pt;
}

auto OverallLineCountingSet::
create(boost::property_tree::ptree const& pt) -> std::unique_ptr<OverallLineCountingSet>
{
	return stdx::make_unique<OverallLineCountingSet>(
			pt.get<std::string>("datasetName"),
			cv::Range{pt.get<int>("frameRange.start"), pt.get<int>("frameRange.end")}
	);
}

auto OverallLineCountingScenario::
evaluate(OverallLineCounter const& counter) const -> vector<FullResult>
{
    LineAndRegionCombiner combiner{counter};

    combiner.trainLineBased(trainings);

    vector<FullResult> results;
    for (auto const& test : tests)
    {
        FullResult thisResult = combiner.predictLineBased(test);
//        thisResult.plot(true,true);
//        thisResult.plot(false,true);

        results.push_back(thisResult);
    }
    return results;
}

auto OverallLineCountingScenario::
evaluateAggregate(OverallLineCounter const& counter) const -> FullResult
{
    FullResult aggregateResult;
    for (auto const& result : evaluate(counter))
    {
        aggregateResult.horizontalAdd(result);
    }
    return aggregateResult;
}

auto OverallLineCountingScenario::
evaluate(LineAndRegionCombiner const& counter) const -> vector<FullResult>
{
    auto combiner = counter.clone();
    combiner->train(trainings);

    vector<FullResult> results;
    for (auto const& test : tests)
    {
        FullResult thisResult = combiner->predictWithCombination(test);
//        thisResult.plot(true,true);
//        thisResult.plot(false,true);

        results.push_back(thisResult);
    }
    return results;
}

auto OverallLineCountingScenario::
evaluateAggregate(LineAndRegionCombiner const& counter) const -> FullResult
{
    FullResult aggregateResult;
    for (auto const& result : evaluate(counter))
    {
        aggregateResult.horizontalAdd(result);
    }
    return aggregateResult;
}


auto OverallLineCountingScenario::
evaluateRegion(RegionCounter const& counter, int nLines, Size size) const -> vector<FullResult>
{
    if (size.area()==0)
    {
        size = config::frames(trainings[0].datasetName)[0].size();
    }

    LineAndRegionCombiner combiner{counter, size};
    combiner.trainRegion(trainings, nLines);

    vector<FullResult> results;
    for (auto const& test : tests)
    {
        FullResult thisResult = combiner.predictRegion(test, nLines);
//        thisResult.plot(true,true);
//        thisResult.plot(false,true);

        results.push_back(thisResult);
    }
    return results;
}

auto OverallLineCountingScenario::
evaluateRegionAggregate(RegionCounter const& counter, int nLines, Size size) const -> FullResult
{
    return FullResult::horizontalMerge(evaluateRegion(counter, nLines, size));
}
