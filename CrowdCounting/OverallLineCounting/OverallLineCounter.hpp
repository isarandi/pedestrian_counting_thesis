#ifndef CROWDCOUNTING_OVERALLLINECOUNTING_OVERALLLINECOUNTER_HPP_
#define CROWDCOUNTING_OVERALLLINECOUNTING_OVERALLLINECOUNTER_HPP_

#include <MachineLearning/Regression.hpp>
#include <CrowdCounting/RegionCounting/RegionCounter.hpp>
#include <CrowdCounting/LineCounting/LineCountingSet.hpp>
#include <cvextra/configfile.hpp>
#include <cvextra/math.hpp>
#include <opencv2/core/core.hpp>
#include <string>
#include <vector>

namespace crowd {
namespace linecounting {

class FullResult;
class LineAndRegionCombiner;

class OverallLineCountingSet
{
public:
	OverallLineCountingSet(
			std::string const& datasetName,
			cv::Range frameRange = cv::Range::all())
		:  datasetName(datasetName), frameRange(frameRange)
		{}

	static auto makeMany(
			std::vector<std::string> const& names,
			cv::Range const& frameRange = cv::Range::all()
			) -> std::vector<OverallLineCountingSet>;

	auto getLineCountingSet(cvx::LineSegment const& seg) const -> LineCountingSet;

	std::string datasetName;
	cv::Range frameRange = cv::Range::all();

	CVX_CONFIG_SINGLE(OverallLineCountingSet)
};

class OverallLineCounter {
public:
	virtual void train(
			std::vector<OverallLineCountingSet> const& trainingSet) = 0;
	virtual auto predict(
			OverallLineCountingSet const& testSet) const -> PredictionWithConfidence2 = 0;

	virtual auto getNumberOfLines() const -> int = 0;
	virtual auto getGroundTruth(OverallLineCountingSet const&) const -> cv::Mat2d = 0;

	virtual ~OverallLineCounter(){}

	CVX_CLONE_IN_BASE(OverallLineCounter)
	CVX_CONFIG_BASE(OverallLineCounter)
};

class OverallLineCountingScenario
{
public:
    std::vector<OverallLineCountingSet> trainings;
    std::vector<OverallLineCountingSet> tests;

    auto evaluate(OverallLineCounter const& counter) const -> std::vector<FullResult>;
    auto evaluateAggregate(OverallLineCounter const& counter) const -> FullResult;

    auto evaluateRegion(RegionCounter const& counter, int nLines, cv::Size size) const -> std::vector<FullResult>;
    auto evaluateRegionAggregate(RegionCounter const& counter, int nLines, cv::Size size) const -> FullResult;

    auto evaluate(LineAndRegionCombiner const& counter) const -> std::vector<FullResult>;
    auto evaluateAggregate(LineAndRegionCombiner const& counter) const -> FullResult;
};

} /* namespace linecounting */
} /* namespace crowd */

#endif /* CROWDCOUNTING_OVERALLLINECOUNTING_OVERALLLINECOUNTER_HPP_ */
