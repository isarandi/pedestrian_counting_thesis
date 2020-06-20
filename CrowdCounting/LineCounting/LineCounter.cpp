#include <boost/range/irange.hpp>
#include <cvextra/coords.hpp>
#include <cvextra/cvret.hpp>
#include <cvextra/improc.hpp>
#include <cvextra/LoopRange.hpp>
#include <cvextra/mats.hpp>
#include <cvextra/visualize.hpp>
#include <CrowdCounting/LineCounting/FeatureSlices.hpp>
#include <CrowdCounting/LineCounting/LineCounter.hpp>
#include <CrowdCounting/LineCounting/SlidingWindow.hpp>
#include <CrowdCounting/LineCounting/WindowBasedCounting.hpp>
#include <CrowdCounting/PersonLocations.hpp>
#include <MachineLearning/LearningSet.hpp>
#include <Persistence.hpp>
#include <Run/config.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/core/mat.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <cmath>
#include <vector>

using namespace std;
using namespace cv;
using namespace cvx;
using namespace crowd;
using namespace crowd::linecounting;

void LineCounter::
train(vector<LineCountingSet> const& sequences)
{
    LearningSet trainingSet;
    //if (!augmentWithMirrored)
    {
        for (LineCountingSet const& sequence : sequences)
        {
            trainingSet.verticalAdd(createLearningSet(sequence, false));
        }
    }

    if (augmentWithMirrored)
    {
        for (LineCountingSet const& sequence : sequences)
        {
            trainingSet.verticalAdd(createLearningSet(sequence, true));
        }
    }

    regression->train(trainingSet);
}

auto LineCounter::
predictPerFrame(
		LineCountingSet const& sequence
        ) const -> PredictionWithConfidence
{
    PredictionWithConfidence pc = predictPerWindow(sequence);

    int nFrames = sequence.loadLocations().getFrameCount();
    return representer.toContinuousSolution(pc, nFrames);
}

auto LineCounter::
predictPerWindow(
        LineCountingSet const& sequence
        ) const -> PredictionWithConfidence
{
    LearningSet testSet = createLearningSet(sequence, false);
    return regression->predictWithConfidence(testSet.X);
}

auto LineCounter::
createLearningSet(
        LineCountingSet const& sequence,
        bool mirrored
        ) const -> LearningSet
{
    string name =
            cvx::configfile::json_str(representer.describe())
            + cvx::configfile::json_str(sequence.describe()) + (mirrored?"mirrored":"");

    auto XY = Persistence::loadOrComputeMats(name, {"X", "Y"}, [&]()
    {
        cout << name << endl;
        LearningSet ls = representer.toLearningSet(
                (mirrored ? sequence.loadSlices().horizontalFlipped() : sequence.loadSlices()),
                sequence.segment,
                sequence.loadLocations());

        if (mirrored)
        {
            ls.Y = cvxret::hconcat(
                    ls.Y.colRange(ls.Y.cols/2, ls.Y.cols).clone(),
                    ls.Y.colRange(0, ls.Y.cols/2).clone());
        }

        return vector<Mat>{ls.X, ls.Y};
    });

    return LearningSet{XY[0], XY[1]};
}

auto LineCounter::
describe() const -> boost::property_tree::ptree
{
	boost::property_tree::ptree pt;
	pt.put_child("regression", regression->describe());
	pt.put_child("representer", representer.describe());
	pt.put("augmentWithMirrored", augmentWithMirrored);
	return pt;
}

auto LineCounter::
create(boost::property_tree::ptree const& pt) -> std::unique_ptr<LineCounter>
{
	return stdx::make_unique<LineCounter>(
			*RegressionWithConfidence::create(pt.get_child("regression")),
			*LineLearningRepresenter::create(pt.get_child("representer")),
			pt.get<bool>("augmentWithMirrored")
	);
}

