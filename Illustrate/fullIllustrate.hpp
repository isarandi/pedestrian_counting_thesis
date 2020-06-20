#ifndef ILLUSTRATE_FULLILLUSTRATE_HPP_
#define ILLUSTRATE_FULLILLUSTRATE_HPP_

#include <CrowdCounting/LineCounting/LineCountingExperiment.hpp>
#include <CrowdCounting/Combination/Combiner.hpp>
#include <cvextra/filesystem.hpp>
#include <CrowdCounting/OverallLineCounting/FullResult.hpp>
#include <CrowdCounting/OverallLineCounting/OverallLineCounter.hpp>
#include <vector>

namespace crowd {
class CountingTestResult;
} /* namespace crowd */


namespace crowd
{

void fullIllustrate(
        crowd::linecounting::OverallLineCountingScenario const& scenario,
        crowd::linecounting::OverallLineCounter const& counter,
        cvx::bpath const& path,
		double illustrationFactor = 1.0);

} /* namespace crowd */

#endif /* ILLUSTRATE_FULLILLUSTRATE_HPP_ */
