#include <cvextra/coords.hpp>
#include <cvextra/core.hpp>
#include <cvextra/io.hpp>
#include <cvextra/LoopRange.hpp>
#include <cvextra/mats.hpp>
#include <cvextra/strings.hpp>
#include <CrowdCounting/RegionCounting/CountingFrame.hpp>
#include <CrowdCounting/CrowdCountingUtils.hpp>
#include <CrowdCounting/RegionCounting/FrameCollection.hpp>
#include <CrowdCounting/RegionCounting/Features/PreprocessedFrame.hpp>
#include <CrowdCounting/RegionCounting/SharedModelRegionCounter.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/core/mat.hpp>
#include <opencv2/core/core.hpp>
#include <Persistence.hpp>
#include <sstream>
#include <string>

using namespace std;
using namespace stdx;
using namespace cv;
using namespace cvx;
using namespace crowd;

void SharedModelRegionCounter::
train(FrameCollection const& samples)
{
    Mat1d inputMat =
            Persistence::loadOrComputeMat(
                    getCaseID(samples),
                    [&]{return buildInputMatrix(samples);});

    regression->train({inputMat, buildOutputMatrix(samples)});
}

Mat1d SharedModelRegionCounter::
predict(FrameCollection const& samples) const
{
    Mat1d inputMat =
            Persistence::loadOrComputeMat(
                    getCaseID(samples),
                    [&]{return buildInputMatrix(samples);});

    Mat cellPredictions = regression->predict(inputMat);
    return cellPredictions.reshape(0, samples.size());
}

auto SharedModelRegionCounter::
predictWithConfidence(FrameCollection const& samples) const -> PredictionWithConfidence
{
    Mat1d inputMat =
            Persistence::loadOrComputeMat(
                    getCaseID(samples),
                    [&]{return buildInputMatrix(samples);});

    RegressionWithConfidence* withConfidence =
    		dynamic_cast<RegressionWithConfidence*>(regression.get());

    if (withConfidence)
    {
    	auto cellPredictions = withConfidence->predictWithConfidence(inputMat);
    	Mat1d reshapedMean = cellPredictions.mean.reshape(0, samples.size());
    	Mat1d reshapedVariance = cellPredictions.variance.reshape(0, samples.size());
    	return {reshapedMean, reshapedVariance};
    } else {
    	Mat prediction = regression->predict(inputMat).reshape(0, samples.size());
    	return {prediction, Mat1d::ones(prediction.size())};
    }
}

auto SharedModelRegionCounter::
canGiveConfidence() const -> bool
{
    return dynamic_cast<RegressionWithConfidence*>(regression.get()) != nullptr;
}

auto SharedModelRegionCounter::
buildInputMatrix(FrameCollection const& samples) const -> Mat
{
    int nFeaturesPerCell = extractor.getFeatureCount();
    int nEstimCells = gridSize.area();

    Mat1d X{
            samples.size()*nEstimCells,
            nFeaturesPerCell+(inverseScaleFeature ? 1 : 0),
            0.0};

    auto gridRectangles = crowd::createGridRectangles(gridSize);

    #pragma omp parallel for
    for (int iFrame = 0; iFrame < samples.size(); ++iFrame)
    {
        cvx::io::statusUpdate(cvx::str::format("Extracting region features, frame #%d", iFrame));

        auto prepFrame = extractor.preprocess(samples[iFrame]);

        for (int iRect : cvx::irange(gridRectangles.size()))
        {
            auto features = extractor.extractFeatures(prepFrame, gridRectangles[iRect]);
            cvx::mats::setRow(X, iFrame*nEstimCells + iRect, features);

            if (inverseScaleFeature)
            {
                Mat1d scaleMap = cvx::extractRelativeRoi(prepFrame.scaleMap, gridRectangles[iRect]);
                double scale = scaleMap(cvx::center(scaleMap));

                X(iFrame*nEstimCells+iRect, nFeaturesPerCell) = 1.0/scale;
            }
        }
    }

    cvx::mats::checkNaN(X, "X_unnormalized");
    return X;
}

auto SharedModelRegionCounter::
buildOutputMatrix(FrameCollection const& samples) const -> Mat
{
    auto gridRectangles = crowd::createGridRectangles(gridSize);
    Mat1d Y(samples.size() * gridSize.area(), 1);

    for (int iFrame : cvx::irange(samples.size()))
    {
        for (int iRect : cvx::irange(gridRectangles.size()))
        {
            Y(iFrame*gridRectangles.size() + iRect, 0) =
                    samples[iFrame].countPeopleInRectangle(gridRectangles[iRect]);
        }
    }

    return Y;
}

auto SharedModelRegionCounter::
getCaseID(FrameCollection const& sequence) const -> string
{
    stringstream ss;
    ss << "shared among all" << sequence.getDescription()
       << extractor.getDescription()
       << gridSize.width << "x" << gridSize.height << " invscale:" << inverseScaleFeature;

    return ss.str();
}

auto SharedModelRegionCounter::
getDescription() const -> string
{
    stringstream ss;
    ss << "Model type: shared for all cells" << endl;
    ss << "Grid: w=" << gridSize.width << " h=" << gridSize.height << endl;
    ss << "Regression:" << endl;
    ss << cvx::str::indentBlock(regression->getDescription()) << endl;
    ss << "Feature extractor: " << endl;
    ss << cvx::str::indentBlock(extractor.getDescription()) << endl;
    ss << "Inverse scale feature: " << inverseScaleFeature;

    return ss.str();
}


auto SharedModelRegionCounter::
describe() const -> boost::property_tree::ptree
{
	boost::property_tree::ptree pt;
	pt.put("type", "SharedModelRegionCounter");
	pt.put("gridSize.width", gridSize.width);
	pt.put("gridSize.height", gridSize.height);
	pt.put_child("extractor", extractor.describe());
	pt.put_child("regression", regression->describe());
	pt.put("inverseScaleFeature", inverseScaleFeature);
	return pt;
}

auto SharedModelRegionCounter::
create(boost::property_tree::ptree const& pt) -> std::unique_ptr<SharedModelRegionCounter>
{
	return stdx::make_unique<SharedModelRegionCounter>(
			cv::Size{pt.get<int>("gridSize.width"), pt.get<int>("gridSize.height")},
			*MultiFeatureExtractor::create(pt.get_child("extractor")),
			*RegressionWithConfidence::create(pt.get_child("regression")),
			pt.get<bool>("inverseScaleFeature")
	);
}

