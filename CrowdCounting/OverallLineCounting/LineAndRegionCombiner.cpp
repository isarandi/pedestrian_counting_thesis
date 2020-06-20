#include <CrowdCounting/OverallLineCounting/LineAndRegionCombiner.hpp>
#include <CrowdCounting/LineCounting/LineCountingExperiment.hpp>
#include <CrowdCounting/RegionCounting/FrameCollection.hpp>
#include <CrowdCounting/CrowdCountingUtils.hpp>
#include <cvextra/core.hpp>
#include <cvextra/gui.hpp>
#include <cvextra/cvret.hpp>
#include <cvextra/mats.hpp>
#include <cvextra/filesystem.hpp>
#include <cvextra/LoopRange.hpp>
#include <cvextra/io.hpp>
#include <CrowdCounting/OverallLineCounting/OverallLineCounter.hpp>
#include <Python/Pyplot.hpp>
#include <CrowdCounting/Combination/Combiner.hpp>
#include <Illustrate/plots.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/core/mat.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <Run/config.hpp>
#include <Persistence.hpp>
#include <set>
#include <string>
#include <vector>

using namespace std;
using namespace cv;
using namespace cvx;
using namespace crowd;
using namespace crowd::linecounting;

namespace crowd {
namespace linecounting {
class LineCountingScenario;
} /* namespace linecounting */
} /* namespace crowd */

void LineAndRegionCombiner::
train(std::vector<OverallLineCountingSet> const& trainingSet)
{
    trainLineBased(trainingSet);
    trainRegion(trainingSet, oLineCounter->getNumberOfLines());
}

void LineAndRegionCombiner::
trainLineBased(std::vector<OverallLineCountingSet> const& trainingSet)
{
    oLineCounter->train(trainingSet);
    if (regionCounter)
    {
        regionCounter->setGridSize(Size{oLineCounter->getNumberOfLines()+1,1});
    }
}

void LineAndRegionCombiner::
trainRegion(std::vector<OverallLineCountingSet> const& trainingSet, int nLines)
{
    regionCounter->setGridSize(Size{nLines+1,1});
    auto trainingFrames = createCountingFrames(trainingSet);
    regionCounter->train(trainingFrames);
}

auto LineAndRegionCombiner::
predictLineBased(
        OverallLineCountingSet const& testSet
        ) const -> FullResult
{
    auto oLinePrediction = oLineCounter->predict(testSet);

    if (regionProcessingSize.area()==0)
    {
        regionProcessingSize = config::frames(testSet.datasetName)[0].size();
    }

    auto testFrames = createCountingFrames({testSet});
    int nLines = oLineCounter->getNumberOfLines();

    FullResult full {
        (Mat1d)cvx::mats::matFromRows(
                crowd::gridQuantizeAll(
                        testFrames.getFrames(),
                        Size{nLines+1,1})
        ).colRange(1, nLines),
        (Mat2d) oLineCounter->getGroundTruth(testSet),
        PredictionWithConfidence{
                Mat1d::zeros(testFrames.size(), nLines-1),
                Mat1d::zeros(testFrames.size(), nLines-1)
        },
        oLinePrediction,
        Mat1d{},
        Mat2d{}
    };

    return full;
}

auto LineAndRegionCombiner::
predictRegion(
        OverallLineCountingSet const& testSet, int nLines
        ) const -> FullResult
{
    auto testFrames = createCountingFrames({testSet});

    auto predictedRegionCounts = regionCounter->predictWithConfidence(testFrames);

    FullResult full {
        (Mat1d)cvx::mats::matFromRows(
                crowd::gridQuantizeAll(
                        testFrames.getFrames(),
                        regionCounter->getGridSize())
        ).colRange(1, nLines),
        Mat2d::zeros(testFrames.size(), nLines),
        PredictionWithConfidence{
                predictedRegionCounts.mean.colRange(1, nLines).clone(),
                predictedRegionCounts.variance.colRange(1, nLines).clone()
        },
        PredictionWithConfidence2{
                        Mat1d::zeros(testFrames.size(), nLines),
                        Mat1d::zeros(testFrames.size(), nLines)},
        Mat1d{},
        Mat2d{}
    };

    return full;
}

auto LineAndRegionCombiner::
predictWithoutCombination(
		OverallLineCountingSet const& testSet
		) const -> FullResult
{
	auto testFrames = createCountingFrames({testSet});
	int nLines = oLineCounter->getNumberOfLines();

	auto predictedRegionCounts = regionCounter->predictWithConfidence(testFrames);

	FullResult full {
		(Mat1d)cvx::mats::matFromRows(
				crowd::gridQuantizeAll(
						testFrames.getFrames(),
						regionCounter->getGridSize())
		).colRange(1, nLines),
		(Mat2d) oLineCounter->getGroundTruth(testSet),
		PredictionWithConfidence{
				predictedRegionCounts.mean.colRange(1, nLines).clone(),
				predictedRegionCounts.variance.colRange(1, nLines).clone()
		},
		oLineCounter->predict(testSet),
		Mat1d{},
		Mat2d{}
	};

	return full;
}

auto LineAndRegionCombiner::
predictWithCombination(
		OverallLineCountingSet const& testSet
		) const -> FullResult
{
	FullResult res = predictWithoutCombination(testSet);
	return combine(res);
}

auto LineAndRegionCombiner::
combine(FullResult const& r) const -> FullResult
{
    cout << "Combining..." << endl;
    int nFrames = r.desiredRegionCounts.rows;

    int nLines = r.desiredLineFlow.cols;

    cvx::mats::checkNaN(r.predictedLineFlow.mean.reshape(1), "predLineMean");

    Mat1d predictedRegionCountsMean = cvxret::medianFilter(r.predictedRegionCounts.mean, Size{1,3});
    Mat1d predictedRegionCountsVariance = cvxret::medianFilter(r.predictedRegionCounts.variance, Size{1,3});

    cv::GaussianBlur(predictedRegionCountsMean, predictedRegionCountsMean, Size{1,9}, 2.0,2.0, BORDER_REPLICATE);
    cv::GaussianBlur(predictedRegionCountsVariance, predictedRegionCountsVariance, Size{1,9}, 3.0,3.0, BORDER_REPLICATE);

    Mat2d predictedLineFlowMean = cvxret::medianFilter(r.predictedLineFlow.mean, Size{1,3});
    Mat2d predictedLineFlowVariance = cvxret::medianFilter(r.predictedLineFlow.variance, Size{1,3});

    cv::GaussianBlur(predictedLineFlowMean, predictedLineFlowMean, Size{1,9}, 2.0,2.0, BORDER_REPLICATE);
    cv::GaussianBlur(predictedLineFlowVariance, predictedLineFlowVariance, Size{1,9}, 3.0,3.0, BORDER_REPLICATE);

	cvx::gui::startTweaking({
        {"lineFlowPenaltyFactor", 0.001, 10, 0.08},
        {"regionPenaltyFactor", 0.0001, 0.1, 0.008},
        {"SORomega", 0.1, 1.99, 0.7},
        {"SORiterations", 100, 10000, 2500},
        {"priorMean", 0, 10, 2},
        {"priorVariance", 0.0, 0.005, 0.001},
        {"rho", 0, 0.5, 0.005}},
	    [&,this](map<string,double> const& p)
        {
            auto improved =
                    crowd::linecounting::solveCombinationCoarseToFine(
                            {r.predictedRegionCounts.mean, r.predictedLineFlow.mean},
                            {r.desiredRegionCounts, r.desiredLineFlow},
                            p.at("lineFlowPenaltyFactor")*cvret::divide(1.0, r.predictedLineFlow.variance),
                            p.at("priorMean"),p.at("priorVariance"),
                            p.at("regionPenaltyFactor")*cvret::divide(1.0, r.predictedRegionCounts.variance),
                            p.at("rho"), p.at("SORomega"), p.at("SORiterations"));

            FullResult r2 = r;
            r2.improvedLineFlow = improved.l;
            r2.improvedRegionCounts = improved.r;

            //r2.plot(true, false).saveAndClose("/work/sarandi/crowd/test_crange_normal_combined.png");
            return r2.plot(true, false).renderAndClose();
        }, true);

    auto improved =
            crowd::linecounting::solveCombinationCoarseToFine(
                    {predictedRegionCountsMean, predictedLineFlowMean},
                    {r.desiredRegionCounts, r.desiredLineFlow},
                    lineFlowPenaltyFactor*cvret::divide(1.0, predictedLineFlowVariance),
					2, 0.0,
					regionPenaltyFactor*cvret::divide(1.0, predictedRegionCountsVariance),
					0.005, SORomega, SORiterations);

    return FullResult{r.desiredRegionCounts, r.desiredLineFlow, r.predictedRegionCounts, r.predictedLineFlow, improved.r, improved.l};
}

auto LineAndRegionCombiner::
createCountingFrames(
        vector<OverallLineCountingSet> const& ls
		) const -> FrameCollection
{
    FrameCollection result;
    set<string> addedDatasetNames;

    Mat1d scaleMap =
    		cvx::io::readDoubleMatFromCSV(
    				config::scaleMapPath(ls[0].datasetName));

    BinaryMat stencil =
    		cvx::imread(config::roiStencilPath(ls[0].datasetName), IMREAD_GRAYSCALE);

    for (auto const& s : ls)
    {
        if (addedDatasetNames.count(s.datasetName))
        {
            continue;
        }
        PersonLocations locations{config::locationPath(s.datasetName)};

        FrameCollection frames =
                FrameCollection{
                    s.datasetName,
                    CountingFrame::allFromFolder(
                            config::framePath(s.datasetName),
                            config::maskPath(s.datasetName),
							config::textonPath(s.datasetName),
                            locations.getRelativePositionsByFrame(stencil.size()),
							regionProcessingSize,
                            scaleMap,
							stencil)
                    }.range(s.frameRange);

        addedDatasetNames.insert(s.datasetName);
        result.append(frames);
    }

    return result;
}

auto LineAndRegionCombiner::
describe() const -> boost::property_tree::ptree
{
	boost::property_tree::ptree pt;
	pt.put_child("oLineCounter", oLineCounter->describe());
	pt.put_child("regionCounter", regionCounter->describe());
	pt.put("regionProcessingSize.width", regionProcessingSize.width);
	pt.put("regionProcessingSize.height", regionProcessingSize.height);
	pt.put("regionPenaltyFactor", regionPenaltyFactor);
	pt.put("lineFlowPenaltyFactor", lineFlowPenaltyFactor);
	pt.put("SORomega", SORomega);
	pt.put("SORiterations", SORiterations);
	return pt;
}

auto LineAndRegionCombiner::
create(boost::property_tree::ptree const& pt) -> std::unique_ptr<LineAndRegionCombiner>
{
	return stdx::make_unique<LineAndRegionCombiner>(
			*OverallLineCounter::create(pt.get_child("extractor")),
			*RegionCounter::create(pt.get_child("regression")),
			Size{pt.get<int>("regionProcessingSize.width"), pt.get<int>("regionProcessingSize.height")},
			pt.get<double>("regionPenaltyFactor"),
			pt.get<double>("lineFlowPenaltyFactor"),
			pt.get<double>("SORomega"),
			pt.get<int>("SORiterations")
	);
}


//pyx::Pyplot plt2;
//for (int iLine : cvx::irange(nLines))
//{
//    Mat1d boxCurveImproved =
//            LineCountingResult::boxEvaluationCurve(
//                    improved.l.col(iLine),
//                    outset.desiredLineFlow.col(iLine));
//    Mat1d boxCurveBefore =
//            LineCountingResult::boxEvaluationCurve(
//                    outset.predictedLineFlow.col(iLine),
//                    outset.desiredLineFlow.col(iLine));
//
//    plotWithStdev(plt2, boxCurveImproved.col(0), boxCurveImproved.col(2), "r");
//    plotWithStdev(plt2, boxCurveImproved.col(1), boxCurveImproved.col(3), "r--");
//    plotWithStdev(plt2, boxCurveBefore.col(0), boxCurveBefore.col(2), "k");
//    plotWithStdev(plt2, boxCurveBefore.col(1), boxCurveBefore.col(3), "k--");
//    plt2.show();
//}
