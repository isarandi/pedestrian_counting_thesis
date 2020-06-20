#include <CrowdCounting/OverallLineCounting/SharedModelOverallLineCounter.hpp>
#include <boost/range/irange.hpp>
#include <cvextra/ImageLoadingIterable.hpp>
#include <cvextra/LoopRange.hpp>
#include <cvextra/mats.hpp>
#include <cvextra/vectors.hpp>
#include <CrowdCounting/LineCounting/LineCountingSet.hpp>
#include <CrowdCounting/PersonLocations.hpp>
#include <MachineLearning/Regression.hpp>
#include <opencv2/core/mat.hpp>
#include <opencv2/core/operations.hpp>
#include <Run/config.hpp>

using namespace crowd::linecounting;
using namespace std;
using namespace cv;
using namespace cvx;

SharedModelOverallLineCounter::
SharedModelOverallLineCounter(
		LineCounter const& lineCounter, int nLines)
	: lineCounter(lineCounter)
	, nLines(nLines)
{

}

auto SharedModelOverallLineCounter::
getLines(Size size) const -> vector<LineSegment>
{
    vector<LineSegment> lines;

    for (int ix : cvx::irange(nLines))
    {
    	double x = (1+ix) * size.width/(nLines+1);
        lines.push_back({{x,0},{x, size.height-1}});
    }
    return lines;
}

void SharedModelOverallLineCounter::
train(std::vector<OverallLineCountingSet> const& trainingSet)
{
	vector<LineSegment> lines =
			getLines(config::frames(trainingSet[0].datasetName)[0].size());

    vector<LineCountingSet> lineCountingSets;
    for (auto const& e : trainingSet)
    {
    	cvx::vectors::push_back_all(lineCountingSets, LineCountingSet::cross({e.datasetName}, lines, e.frameRange));
    }
    lineCounter.train(lineCountingSets);
}

auto SharedModelOverallLineCounter::
predict(OverallLineCountingSet const& testSet) const -> PredictionWithConfidence2
{
	vector<LineSegment> lines =
			getLines(config::frames(testSet.datasetName)[0].size());

	PredictionWithConfidence2 pc;
	for (auto const& line : lines)
	{
		auto pcPart = lineCounter.predictPerFrame(crowd::linecounting::LineCountingSet{testSet.datasetName, line, testSet.frameRange});
		cvx::hconcat(pc.mean, pcPart.mean.reshape(2), pc.mean);
		cvx::hconcat(pc.variance, pcPart.variance.reshape(2), pc.variance);
	}

	return pc;
}

auto SharedModelOverallLineCounter::
getNumberOfLines() const -> int
{
	return nLines;
}

auto SharedModelOverallLineCounter::
getGroundTruth(OverallLineCountingSet const& testSet) const -> cv::Mat2d
{
	vector<LineSegment> lines =
			getLines(config::frames(testSet.datasetName)[0].size());

	Mat1d result;
	for (auto const& line : lines)
	{
		auto testSequence = LineCountingSet{testSet.datasetName, line, testSet.frameRange};
		Mat1d desiredPerFrame = testSequence.loadLocations().getInstantFlow(line);
		cvx::hconcat(result, desiredPerFrame, result);
	}

	return result.reshape(2);
}

auto SharedModelOverallLineCounter::
describe() const -> boost::property_tree::ptree
{
	boost::property_tree::ptree pt;
	pt.put("type", "SharedModelOverallLineCounter");
	pt.put("nLines", nLines);
	pt.put_child("lineCounter", lineCounter.describe());
	return pt;
}

auto SharedModelOverallLineCounter::
create(boost::property_tree::ptree const& pt) -> std::unique_ptr<SharedModelOverallLineCounter>
{
	return stdx::make_unique<SharedModelOverallLineCounter>(
			*LineCounter::create(pt.get_child("lineCounter")),
			pt.get<int>("nLines")
	);
}
