#ifndef FLOWCOUNTER_H
#define FLOWCOUNTER_H

#include <CrowdCounting/LineCounting/SlidingWindow.hpp>
#include <CrowdCounting/LineCounting/FeatureSlices.hpp>
#include <CrowdCounting/LineCounting/WindowBasedCounting.hpp>
#include <CrowdCounting/LineCounting/LineCountingSet.hpp>
#include <CrowdCounting/LineCounting/LineLearningRepresenter.hpp>
#include <CrowdCounting/PersonLocations.hpp>
#include <MachineLearning/Regression.hpp>
#include <opencv2/core/core.hpp>
#include <stdx/cloning.hpp>
#include <vector>
#include <boost/property_tree/ptree.hpp>

namespace crowd {
namespace linecounting {

class LineCounter
{
public:
    LineCounter(
            RegressionWithConfidence const& regression,
			LineLearningRepresenter const& representer,
			bool augmentWithMirrored)
        : regression(regression.clone())
		, representer(representer)
        , augmentWithMirrored(augmentWithMirrored)
    {}

    void train(std::vector<LineCountingSet> const& sequences);

    auto predictPerFrame(LineCountingSet const& sequence) const -> PredictionWithConfidence;
    auto predictPerWindow(LineCountingSet const& sequence) const -> PredictionWithConfidence;

    auto getRepresenter() const -> LineLearningRepresenter const& {return representer;}
    auto getRegression() const -> RegressionWithConfidence const& {return *regression;}

    auto createLearningSet(
            LineCountingSet const& sequence,
            bool mirrored
            ) const -> LearningSet;

    CVX_CLONE_IN_SINGLE(LineCounter)
    CVX_CONFIG_DERIVED(LineCounter)

	virtual ~LineCounter(){}

private:
    stdx::cloned_unique_ptr<RegressionWithConfidence> regression;
    LineLearningRepresenter representer;
    bool augmentWithMirrored;


};


}



}

#endif // FLOWCOUNTER_H
