#include <boost/range/irange.hpp>
#include <cvextra/io.hpp>
#include <cvextra/LoopRange.hpp>
#include <cvextra/mats.hpp>
#include <cvextra/strings.hpp>
#include <CrowdCounting/RegionCounting/CountingFrame.hpp>
#include <CrowdCounting/CrowdCountingUtils.hpp>
#include <CrowdCounting/RegionCounting/AllToAllRegionCounter.hpp>
#include <CrowdCounting/RegionCounting/FrameCollection.hpp>
#include <CrowdCounting/RegionCounting/Features/PreprocessedFrame.hpp>
#include <opencv2/core/mat.hpp>
#include <opencv2/core/core.hpp>
#include <Persistence.hpp>
#include <iostream>
#include <sstream>
#include <vector>
#include <omp.h>

using namespace std;
using namespace stdx;
using namespace cv;
using namespace cvx;
using namespace crowd;

void AllToAllRegionCounter::
train(FrameCollection const& samples)
{
    Mat1d inputMat =
            Persistence::loadOrComputeMat(
                    getCaseID(samples),
                    [&]{return buildInputMatrix(samples);});

    regression->train({inputMat, buildOutputMatrix(samples)});
}

auto AllToAllRegionCounter::
predict(FrameCollection const& samples) const -> Mat1d
{
    Mat1d inputMat =
            Persistence::loadOrComputeMat(
                    getCaseID(samples),
                    [&]{return buildInputMatrix(samples);});

    return regression->predict(inputMat);
}

auto AllToAllRegionCounter::
predictWithConfidence(
		FrameCollection const& samples
		) const -> PredictionWithConfidence
{
    Mat1d inputMat =
            Persistence::loadOrComputeMat(
                    getCaseID(samples),
                    [&]{return buildInputMatrix(samples);});

    RegressionWithConfidence* withConfidence =
    		dynamic_cast<RegressionWithConfidence*>(regression.get());

    if (withConfidence)
    {
    	return withConfidence->predictWithConfidence(inputMat);
    } else {
    	Mat1d prediction = regression->predict(inputMat);
    	return {prediction, Mat1d::ones(prediction.size())};
    }

}

auto AllToAllRegionCounter::
canGiveConfidence() const -> bool
{
    return dynamic_cast<RegressionWithConfidence*>(regression.get()) != nullptr;
}

auto AllToAllRegionCounter::
buildInputMatrix(FrameCollection const& samples) const -> Mat1d
{
    cout << "Building input matrix for all-to-all ..." << endl;

    int nFeaturesPerCell = extractor.getFeatureCount();
    Mat1d X{samples.size(), gridSize.area() * nFeaturesPerCell};

    auto gridRectangles = crowd::createGridRectangles(gridSize);

    #pragma omp parallel for
    for (int iFrame = 0; iFrame < samples.size(); ++iFrame)
    {
        cvx::io::statusUpdate(cvx::str::format("Extracting features (all-to-all) from frame #%d", iFrame));
        auto prepFrame = extractor.preprocess(samples[iFrame]);

        for (int iRect : cvx::irange(gridRectangles.size()))
        {
            auto features = extractor.extractFeatures(prepFrame, gridRectangles[iRect]);
            cvx::mats::setRow(X, iFrame, iRect*nFeaturesPerCell, features);
        }
    }
    cvx::mats::checkNaN(X, "X_unnormalized");

    cout << "Done building input matrix!" << endl;
    return X;
}

auto AllToAllRegionCounter::
buildOutputMatrix(FrameCollection const& samples) const -> Mat1d
{
    auto gridRectangles = crowd::createGridRectangles(gridSize);
    Mat1d Y(samples.size(), gridRectangles.size());

    for (Point p : cvx::points(Y))
    {
        Y(p) = samples[p.y].countPeopleInRectangle(gridRectangles[p.x]);
    }
    return Y;
}

auto AllToAllRegionCounter::
getCaseID(FrameCollection const& sequence) const -> string
{
    stringstream ss;
    ss << "allToAll" << sequence.getDescription()
       << extractor.getDescription()
       << gridSize.width << "x" << gridSize.height;
    return ss.str();
}

auto AllToAllRegionCounter::
getDescription() const -> string
{
    stringstream ss;
    ss << "Model type: all to all" << endl;
    ss << "Grid: w=" << gridSize.width << " h=" << gridSize.height << endl;
    ss << "Regression:" << endl;
    ss << cvx::str::indentBlock(regression->getDescription()) << endl;
    ss << "Feature extractor: " << endl;
    ss << cvx::str::indentBlock(extractor.getDescription());

    return ss.str();
}


auto AllToAllRegionCounter::
describe() const -> boost::property_tree::ptree
{
	boost::property_tree::ptree pt;
	pt.put("type", "AllToAllRegionCounter");
	pt.put("gridSize.width", gridSize.width);
	pt.put("gridSize.height", gridSize.height);
	pt.put_child("extractor", extractor.describe());
	pt.put_child("regression", regression->describe());
	return pt;
}

auto AllToAllRegionCounter::
create(boost::property_tree::ptree const& pt) -> std::unique_ptr<AllToAllRegionCounter>
{
	return stdx::make_unique<AllToAllRegionCounter>(
			cv::Size{pt.get<int>("gridSize.width"), pt.get<int>("gridSize.height")},
			*MultiFeatureExtractor::create(pt.get_child("extractor")),
			*RegressionWithConfidence::create(pt.get_child("regression"))
	);
}
