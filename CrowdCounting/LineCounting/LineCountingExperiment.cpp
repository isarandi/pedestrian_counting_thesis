#include <CrowdCounting/LineCounting/FeatureSlices.hpp>
#include <CrowdCounting/LineCounting/WindowBasedCounting.hpp>
#include <MachineLearning/LearningSet.hpp>
#include <CrowdCounting/LineCounting/LineCountingExperiment.hpp>
#include <Run/config.hpp>

#include <Python/EasyObject.hpp>
#include <Python/Pyplot.hpp>
#include <Python/PythonEnvironment.hpp>

#include <cvextra/cvenums.hpp>
#include <cvextra/cvret.hpp>
#include <cvextra/io.hpp>
#include <cvextra/visualize.hpp>
#include <cvextra/LoopRange.hpp>
#include <cvextra/mats.hpp>
#include <stdx/stdx.hpp>
#include <stdx/cloning.hpp>
#include <opencv2/core/core.hpp>
#include <boost/range/irange.hpp>
#include <CrowdCounting/LineCounting/LineCounter.hpp>
#include <cmath>
#include <vector>
#include <limits>

using namespace std;
using namespace cv;
using namespace cvx;
using namespace crowd;
using namespace crowd::linecounting;

auto LineCountingTestCase::
run() -> vector<LineCountingResult>
{
    flowCounter.train(scenario.trainings);

    vector<LineCountingResult> results;
    for (LineCountingSet const& testSequence : scenario.tests)
    {
    	PersonLocations locations = testSequence.loadLocations();
    	int nFrames = locations.getFrameCount();

        PredictionWithConfidence predictionPerWindow =
        		flowCounter.predictPerWindow(testSequence);
        Mat1d desiredPerWindow =
        		flowCounter.getRepresenter()
				.createTimeWindowOutputs(locations, testSequence.segment, nFrames);

        PredictionWithConfidence predictionPerFrame =
        		flowCounter.getRepresenter()
				.toContinuousSolution(predictionPerWindow, nFrames);
        Mat1d desiredPerFrame = testSequence.loadLocations().getInstantFlow(testSequence.segment);

        results.push_back(
                LineCountingResult{
                    testSequence,
					desiredPerFrame,
					desiredPerWindow,
                    predictionPerFrame,
                    predictionPerWindow});
    }
    return results;
}

auto LineCountingResult::
boxEvaluationCurve() const -> Mat1d
{
    int nSizes = predictionPerFrame.mean.rows-1;
    Mat1d means{nSizes, 2};
    Mat1d stdevs{nSizes, 2};

    Mat1d cumulativeDifference = cvxret::cumsum(predictionPerFrame.mean-desiredPerFrame);

    for (int windowSize : cvx::irange(nSizes))
    {
        Mat1d absWindowErrors = cv::abs(
                cumulativeDifference.rowRange(windowSize, cumulativeDifference.rows) -
                cumulativeDifference.rowRange(0, cumulativeDifference.rows-windowSize));

        Mat meandst = means.row(windowSize);
        cv::reduce(absWindowErrors, meandst, 0, REDUCE_AVG);

        Mat variancedst = stdevs.row(windowSize);
        cvx::variance(absWindowErrors, variancedst, 0, SRC_TYPE, meandst);
    }
    cv::sqrt(stdevs, stdevs);

    return cvxret::hconcat(means, stdevs);
}

auto LineCountingResult::
boxEvaluationCurve(
        cv::Mat2d prediction,
        cv::Mat2d validation) -> Mat1d
{
    LineCountingResult lineResult;
    lineResult.predictionPerFrame.mean = Mat(prediction.clone()).reshape(1);
    lineResult.desiredPerFrame = Mat(validation.clone()).reshape(1);

    return lineResult.boxEvaluationCurve();
}

auto LineCountingScenario::
describe() const -> boost::property_tree::ptree
{
	boost::property_tree::ptree pt;
	pt.add_child("trainings", cvx::configfile::describeCollection(trainings));
	pt.add_child("tests", cvx::configfile::describeCollection(tests));
	return pt;
}

auto LineCountingScenario::
create(boost::property_tree::ptree const& pt) -> std::unique_ptr<LineCountingScenario>
{
	return stdx::make_unique<LineCountingScenario>(
			cvx::configfile::loadVector<LineCountingSet>(pt.get_child("trainings")),
			cvx::configfile::loadVector<LineCountingSet>(pt.get_child("tests"))
	);
}

auto crowd::linecounting::LineCountingResult::
getDriftSigma() const -> double
{
	Mat1d box = boxEvaluationCurve();
	cv::reduce(box, box, 1, REDUCE_AVG);
	Mat1d sqrtWindowSize = cvret::sqrt(cvxret::cumsum(Mat1d::ones(box.size()))-1);


//	pyx::Pyplot::quickPlot(box);
	Mat1d drift = box/sqrtWindowSize;
//	pyx::Pyplot::quickPlot(drift);
//	pyx::Pyplot::quickPlot(drift.rowRange(50, drift.rows));
	return cv::mean(drift.rowRange(50, drift.rows))[0];
}
//auto LineCountingResult::
//describe() const -> boost::property_tree::ptree
//{
//	boost::property_tree::ptree pt;
//	pt.put("countingSet", countingSet->describe());
//	pt.put("desiredPerFrame", cvx::configfile::saveMat(desiredPerFrame));
//	pt.put("desiredPerWindow", cvx::configfile::saveMat(desiredPerWindow));
//	pt.put("predictionPerFrame.mean", cvx::configfile::saveMat(predictionPerFrame.mean));
//	pt.put("predictionPerFrame.variance", cvx::configfile::saveMat(predictionPerFrame.variance));
//	pt.put("predictionPerWindow.mean", cvx::configfile::saveMat(predictionPerWindow.mean));
//	pt.put("predictionPerWindow.variance", cvx::configfile::saveMat(predictionPerWindow.variance));
//	return pt;
//}
//
//auto LineCountingResult::
//create(boost::property_tree::ptree const& pt) -> std::unique_ptr<LineCountingResult>
//{
//	return stdx::make_unique<LineCountingResult>(
//			cvx::configfile::loadVector<LineCountingSet>(pt.get_child("trainings")),
//			cvx::configfile::loadVector<LineCountingSet>(pt.get_child("tests"))
//	);
//}
