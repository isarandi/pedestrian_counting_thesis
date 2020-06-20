#ifndef CROWDCOUNTING_OVERALLLINECOUNTING_SHAREDMODELOVERALLLINECOUNTER_HPP_
#define CROWDCOUNTING_OVERALLLINECOUNTING_SHAREDMODELOVERALLLINECOUNTER_HPP_

#include <cvextra/math.hpp>
#include <CrowdCounting/LineCounting/LineCounter.hpp>
#include <CrowdCounting/OverallLineCounting/OverallLineCounter.hpp>
#include <opencv2/core/core.hpp>
#include <vector>

namespace crowd {
namespace linecounting {

class SharedModelOverallLineCounter : public OverallLineCounter
{
public:
	SharedModelOverallLineCounter(
			LineCounter const& lineCounter,
			int nLines);

	virtual void train(
			std::vector<OverallLineCountingSet> const& trainingSet);
	virtual auto predict(
			OverallLineCountingSet const& testSet) const -> PredictionWithConfidence2;

	virtual auto getGroundTruth(OverallLineCountingSet const&) const -> cv::Mat2d;
	virtual auto getNumberOfLines() const -> int;

	auto getLineCounter() -> LineCounter const& {return lineCounter;}

	auto getLines(cv::Size size) const -> std::vector<cvx::LineSegment>;

	CVX_CLONE_IN_DERIVED(SharedModelOverallLineCounter)
	CVX_CONFIG_DERIVED(SharedModelOverallLineCounter)

private:
	LineCounter lineCounter;
	int nLines;

};

} /* namespace linecounting */
} /* namespace crowd */

#endif /* CROWDCOUNTING_OVERALLLINECOUNTING_SHAREDMODELOVERALLLINECOUNTER_HPP_ */
