#ifndef RUN_LINECOUNTING_RUNCOMBINATION_H_
#define RUN_LINECOUNTING_RUNCOMBINATION_H_

#include <opencv2/core/core.hpp>
#include <string>


namespace crowd {
namespace linecounting {
class LineCountingScenario;
}
} /* namespace crowd */

namespace crowd {

class CountingOutset
{
public:
    cv::Mat1d desiredRegionCounts;
    cv::Mat1d regionPredictedRegionCounts;
    cv::Mat2d predictedLineFlow;
    cv::Mat2d predictedLineFlowVariance;
    cv::Mat2d desiredLineFlow;
    cv::Mat1d regionPredictionVariance;

    void loadOrCreate(
    		std::string const& name,
			crowd::linecounting::LineCountingScenario const& scenario,
			bool forceCreation);
};

namespace run{
    void testCombination();
}

}

#endif /* RUN_LINECOUNTING_RUNCOMBINATION_H_ */
