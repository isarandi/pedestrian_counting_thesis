#ifndef CROWDCOUNTING_OVERALLLINECOUNTING_LINEANDREGIONCOMBINER_HPP_
#define CROWDCOUNTING_OVERALLLINECOUNTING_LINEANDREGIONCOMBINER_HPP_

#include <CrowdCounting/OverallLineCounting/OverallLineCounter.hpp>
#include <CrowdCounting/RegionCounting/RegionCounter.hpp>
#include <MachineLearning/Regression.hpp>
#include <CrowdCounting/OverallLineCounting/FullResult.hpp>
#include <opencv2/core/core.hpp>
#include <cvextra/filesystem.hpp>
#include <stdx/cloning.hpp>
#include <string>
#include <vector>

namespace crowd {
namespace linecounting {
class LineCountingScenario;
} /* namespace linecounting */
} /* namespace crowd */
namespace pyx {
class Pyplot;
} /* namespace pyx */

namespace crowd {
namespace linecounting {


class LineAndRegionCombiner
{
public:
	LineAndRegionCombiner(
			OverallLineCounter const& oLineCounter,
			RegionCounter const& regionCounter,
			cv::Size regionProcessingSize,
			double regionPenaltyFactor,
			double lineFlowPenaltyFactor,
			double SORomega,
			int SORiterations
			)
		: oLineCounter(oLineCounter.clone())
		, regionCounter(regionCounter.clone())
		, regionProcessingSize(regionProcessingSize)
		, regionPenaltyFactor(regionPenaltyFactor)
		, lineFlowPenaltyFactor(lineFlowPenaltyFactor)
		, SORomega(SORomega)
		, SORiterations(SORiterations)
	{}

    explicit
    LineAndRegionCombiner(RegionCounter const& regionCounter, cv::Size regionProcessingSize)
        : regionCounter(regionCounter.clone())
        , regionProcessingSize(regionProcessingSize)
    {}

    explicit
    LineAndRegionCombiner(OverallLineCounter const& oLineCounter)
        : oLineCounter(oLineCounter.clone())
        , regionProcessingSize(cv::Size{0,0})
    {}

	void train(std::vector<OverallLineCountingSet> const& trainingSet);

	void trainLineBased(std::vector<OverallLineCountingSet> const& trainingSet);
	void trainRegion(std::vector<OverallLineCountingSet> const& trainingSet, int nLines);

	auto predictLineBased(OverallLineCountingSet const& testSet) const -> FullResult;
	auto predictRegion(OverallLineCountingSet const& testSet, int nLines) const -> FullResult;

	auto predictWithoutCombination(OverallLineCountingSet const& testSet) const -> FullResult;
	auto predictWithCombination(OverallLineCountingSet const& testSet) const -> FullResult;

	auto combine(FullResult const& r) const -> FullResult;

	CVX_CONFIG_SINGLE(LineAndRegionCombiner)
	CVX_CLONE_IN_SINGLE(LineAndRegionCombiner)

private:

	auto createCountingFrames(std::vector<OverallLineCountingSet> const& ls) const -> FrameCollection;

	stdx::cloned_unique_ptr<OverallLineCounter> oLineCounter;
	stdx::cloned_unique_ptr<RegionCounter> regionCounter;

	mutable cv::Size regionProcessingSize;

	double regionPenaltyFactor;
	double lineFlowPenaltyFactor;
	double SORomega;
	int SORiterations;
};


} /* namespace linecounting */
} /* namespace crowd */

#endif /* CROWDCOUNTING_OVERALLLINECOUNTING_LINEANDREGIONCOMBINER_HPP_ */
