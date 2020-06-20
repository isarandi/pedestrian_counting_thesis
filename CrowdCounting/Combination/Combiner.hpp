#ifndef CROWDCOUNTING_LineCounting_COMBINER_HPP_
#define CROWDCOUNTING_LineCounting_COMBINER_HPP_

#include <CrowdCounting/LineCounting/LineCounter.hpp>
#include <CrowdCounting/RegionCounting/RegionCounter.hpp>
#include <opencv2/core/core.hpp>
#include <stdx/cloning.hpp>
#include <string>
#include <vector>

namespace crowd {
class RegionCounter;
} /* namespace crowd */

namespace crowd { namespace linecounting {

class RegionAndLineCounts {
public:
    cv::Mat1d r;
    cv::Mat2d l;
};

auto aggregateLinesToRegionsInbetween(cv::Mat2d const& c) -> cv::Mat1d;

class OptimizationState{
public:
    cv::Mat2d c;
    cv::Mat1d a;
};

auto solveCombinationOnSingleScale(
        RegionAndLineCounts const& o,
        RegionAndLineCounts const& g,
        OptimizationState const& initialState,
        cv::Mat2d const& alpha,
		double aprioriOutput,
		double aprioriDeviationReward,
		cv::Mat1d const& gamma,
        double rho,
        double SORomega,
        int nSORiterations
        ) -> OptimizationState;

auto solveCombinationCoarseToFine(
        RegionAndLineCounts o,
        RegionAndLineCounts const& g,
        cv::Mat2d const& alpha,
		double aprioriOutput,
		double aprioriDeviationReward,
		cv::Mat1d const& gamma,
        double rho,
        double SORomega,
        int nSORiterations
        ) -> RegionAndLineCounts;

} /* namespace linecounting */
} /* namespace crowd */

#endif /* CROWDCOUNTING_LineCounting_COMBINER_HPP_ */
