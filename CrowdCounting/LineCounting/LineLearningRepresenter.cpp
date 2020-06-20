#include <CrowdCounting/LineCounting/LineLearningRepresenter.hpp>
#include <cvextra/cvenums.hpp>
#include <cvextra/cvret.hpp>
#include <cvextra/improc.hpp>
#include <cvextra/LoopRange.hpp>
#include <cvextra/math.hpp>
#include <cvextra/visualize.hpp>
#include <cvextra/gui.hpp>
#include <cvextra/mats.hpp>
#include <CrowdCounting/LineCounting/FeatureSlices.hpp>
#include <CrowdCounting/LineCounting/WindowBasedCounting.hpp>
#include <MachineLearning/LearningSet.hpp>
#include <opencv2/core/mat.hpp>
#include <opencv2/core/operations.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <stdx/stdx.hpp>
#include <vector>
#include <omp.h>

using namespace std;
using namespace cv;
using namespace cvx;
using namespace crowd;
using namespace crowd::linecounting;

crowd::linecounting::LineLearningRepresenter::
LineLearningRepresenter(
		LineMultiFeatureExtractor const& extractor,
		SlidingWindow<int> frameWindow,
		SlidingWindow<double> relativeSectionWindow,
		int nOutputSections,
		double flowSegmentationThreshold,
		double radius,
		int cancellationDistance)
	: extractor(extractor)
	, frameWindow(frameWindow)
	, relativeSectionWindow(relativeSectionWindow)
	, nOutputSections(nOutputSections)
	, flowSegmentationThreshold(flowSegmentationThreshold)
    , radius(radius)
    , cancellationDistance(cancellationDistance)
{

}

crowd::linecounting::LineLearningRepresenter::
LineLearningRepresenter(
        LineMultiFeatureExtractor const& extractor,
        SlidingWindow<int> frameWindow,
        int nMicroSections,
        int sectionSizeInMicroSections,
        int sectionStepInMicroSections,
        int nOutputSections,
        double flowSegmentationThreshold,
        double radius,
        int cancellationDistance)
    : extractor(extractor)
    , frameWindow(frameWindow)
    , nOutputSections(nOutputSections)
    , flowSegmentationThreshold(flowSegmentationThreshold)
    , radius(radius)
    , cancellationDistance(cancellationDistance)
{
    double microSectionSize = 1.0/static_cast<double>(nMicroSections);
    double sectionStep = std::min(1.0, static_cast<double>(sectionStepInMicroSections) * microSectionSize);
    double sectionSize = std::min(1.0, static_cast<double>(sectionSizeInMicroSections) * microSectionSize);
    relativeSectionWindow = SlidingWindow<double>{sectionSize, sectionStep};
}

auto crowd::linecounting::LineLearningRepresenter::
segmentMovement(
        Mat2d const& flowSlice,
        Mat3b const& imageSlice
		) const -> Mat1b
{
    return segmentMovementStatic(flowSlice, imageSlice, flowSegmentationThreshold);
}


auto crowd::linecounting::LineLearningRepresenter::
segmentMovementStatic(
        Mat2d const& flowSlice,
        Mat3b const& imageSlice,
		double threshold
		) -> Mat1b
{
    Mat1f flowX = cvret::extractChannel(flowSlice, 0).clone();
    Mat1b result = Mat1b::zeros(flowSlice.size());
    result.setTo((int)CrossingDir::LEFTWARD, flowX < -threshold);
    result.setTo((int)CrossingDir::RIGHTWARD, flowX > threshold);

    return result;
}

auto crowd::linecounting::LineLearningRepresenter::
toLearningSet(
		FeatureSlices slices,
		LineSegment const& targetSegment,
		PersonLocations const& locations
		) const -> LearningSet
{
    //--- Blur the flow
    Mat1f flowX = cvret::extractChannel(slices.flow, 0);
    Mat1f flowY = cvret::extractChannel(slices.flow, 1);

//    cvx::gui::startTweaking({
//        {"GaussT", 0, 1, 0.02},
//        {"GaussS", 0, 5, 1.1},
//        {"MedianT", 1, 10, 3},
//        {"MedianS", 1, 10, 3},
//        {"threshold", 0, 10, 1.00}},
//        [&](map<string,double> const& params)
//        {
//            Mat1f flowX2 =
//                    cvxret::medianFilter(
//                            flowX,
//                            Size{(int)params.at("MedianT"),(int)params.at("MedianS")});
//
//            cv::GaussianBlur(flowX2, flowX2, Size{5,5}, params.at("GaussT"), params.at("GaussS"));
//
//            Mat1b directionLabels =
//                    segmentMovementStatic(
//                            cvret::merge({flowX2,flowY}),
//                            slices.image,
//                            params.at("threshold"));
//
//            return cvxret::vconcat(
//                cvxret::resizeByRatio(cvx::visu::maskIllustration(slices.image, directionLabels==1),0.5).t(),
//                cvxret::resizeByRatio(cvx::visu::maskIllustration(slices.image, directionLabels==2),0.5).t());
//        }
//    );

//////
// FOR CRANGE_TOP:
    flowX = cvxret::medianFilter(flowX, Size{3,3});
    cv::GaussianBlur(flowX, flowX, Size{5,5}, 0.02, 1.1);
    cvx::setChannel(Mat1d{flowX}, slices.flow, 0);
    slices.directionLabels = segmentMovement(cvret::merge({flowX,flowY}), slices.image);
///////

////////
// FOR UCSD VIDD
//    cv::GaussianBlur(flowX, flowX, Size{5,5}, 0.02, 2.7);
//    cvx::setChannel(Mat1d{flowX}, slices.flow, 0);
//    slices.directionLabels = segmentMovement(cvret::merge({flowX,flowY}), slices.image);
////

    Mat1d input = createTimeWindowDescriptors(slices, targetSegment);
    Mat1d output = createTimeWindowOutputs(locations, targetSegment, slices.size().height);
    return LearningSet{input, output};
}

auto crowd::linecounting::LineLearningRepresenter::
toContinuousSolution(
		PredictionWithConfidence const& pc,
		int nFrames) const-> PredictionWithConfidence
{
	auto mean = crowd::linecounting::windowEstimatesToDeltasBoxFilter(pc.mean, frameWindow, nFrames);
	auto vari = crowd::linecounting::windowEstimatesToDeltasBoxFilter(pc.variance, frameWindow, nFrames);

	auto reducedMean = cvret::hconcat(
			cvret::reduce(mean.colRange(0,nOutputSections), 1, REDUCE_SUM),
			cvret::reduce(mean.colRange(nOutputSections, mean.cols), 1, REDUCE_SUM));

	auto reducedVariance = cvret::hconcat(
			cvret::reduce(vari.colRange(0,nOutputSections), 1, REDUCE_SUM),
			cvret::reduce(vari.colRange(nOutputSections, vari.cols), 1, REDUCE_SUM));

	return PredictionWithConfidence{reducedMean, reducedVariance};
}

auto crowd::linecounting::LineLearningRepresenter::
createTimeWindowDescriptors(
        FeatureSlices const& slices,
        LineSegment const& targetSegment
        ) const -> Mat1d
{
    Mat1d result;
    SlidingWindow<double> absoluteSectionWindow =
                targetSegment.length() * relativeSectionWindow;

//    cv::imshow("im", slices.image.t());
//    cv::imshow("c", slices.canny.t());
//    cv::imshow("f", slices.foregroundMask.t());
//    cv::imshow("flow", cvx::visu::vectorFieldAsHSVAsBGR(slices.flow).t());
//    cv::imshow("texton", cvx::visu::labels(slices.textonMap).t());
//    cv::imshow("directions", cvx::visu::labels(slices.directionLabels).t());
//    cvx::waitKey(' ');


    int nSections = relativeSectionWindow.sectionCountOver(1.0);
    for (int iSection : cvx::irange(nSections))
    {
        auto currentSectionWindow = absoluteSectionWindow.ith(iSection);
        auto sectionInput =
                createSingleSectionWindowDescriptors(
                        slices(Range::all(), cv::Range(currentSectionWindow)));

        cvx::hconcat(result, sectionInput, result);
    }

    return result;
}

auto crowd::linecounting::LineLearningRepresenter::
createTimeWindowOutputs(
		PersonLocations const& locations,
        LineSegment const& targetSegment,
		int nFrames
        ) const -> Mat1d
{
	int nWindows = frameWindow.sectionCountOver(nFrames);
	Mat1d output{nWindows, nOutputSections*2};

	double sectionSegLength = targetSegment.length()/nOutputSections;
	for (int iOutputSection : cvx::irange(nOutputSections))
	{
		LineSegment segPart{
			targetSegment.p1 + targetSegment.dir()*iOutputSection*sectionSegLength,
			targetSegment.p1 + targetSegment.dir()*(iOutputSection+1)*sectionSegLength};

		Mat1d instantFlow =
		            locations.betweenFrames({0,nFrames})
		                    .getInstantFlow(segPart, radius, cancellationDistance);

		for (int iWindow : cvx::irange(nWindows))
		{
			Range windowRange = Range(frameWindow.ith(iWindow));
			Mat1d totalFlowInWindow = cvret::reduce(instantFlow.rowRange(windowRange), 0, REDUCE_SUM);
			Mat1d perFrameAverageFlow = totalFlowInWindow / windowRange.size();

			output(iWindow, iOutputSection) = perFrameAverageFlow(0);
			output(iWindow, nOutputSections+iOutputSection) = perFrameAverageFlow(1);
		}
	}

//    Mat1d instantFlow =
//            locations.betweenFrames({0,nFrames})
//                    .getInstantFlow(targetSegment);
//
//    int nWindows = frameWindow.sectionCountOver(instantFlow.rows);
//    Mat1d output{nWindows, instantFlow.cols};
//
//    for (int iWindow : cvx::irange(nWindows))
//    {
//        Range windowRange = Range(frameWindow.ith(iWindow));
//        Mat1d totalFlowInWindow = cvret::reduce(instantFlow.rowRange(windowRange), 0, REDUCE_SUM);
//        Mat1d perFrameAverageFlow = totalFlowInWindow / windowRange.size();
//
//        perFrameAverageFlow.copyTo(stdx::tempref(output.row(iWindow)));
//    }

    return output;
}

auto crowd::linecounting::LineLearningRepresenter::
createSingleSectionWindowDescriptors(
        FeatureSlices const& slices
        ) const -> Mat1d
{
	int nTimeWindows = frameWindow.sectionCountOver(slices.image.rows);

    Mat1d learningInput{0,1};

    int nWindows = frameWindow.sectionCountOver(slices.image.rows);

    std::vector<Mat1d> results(nWindows);

	#pragma omp parallel for
    for (int iWindow = 0; iWindow < nWindows; ++iWindow)
    {
        Range windowRange = Range(frameWindow.ith(iWindow));
        FeatureSlices windowSlice = slices(windowRange, Range::all());

        results[iWindow] = cvxret::hconcatAll(std::vector<Mat1d>{
            extractor.extractFeatures(
            		windowSlice,
					windowSlice.directionLabels==1),
			extractor.extractFeatures(
					windowSlice,
					windowSlice.directionLabels==2),
//			extractor.extractFeatures(
//					windowSlice,
//					cv::min(Mat(windowSlice.directionLabels==0), windowSlice.foregroundMask))
        });
    }

    return cvxret::vconcatAll(results);
}

auto crowd::linecounting::LineLearningRepresenter::
describe() const -> boost::property_tree::ptree
{
	boost::property_tree::ptree pt;
	pt.put_child("extractor", extractor.describe());
	pt.put("frameWindow_size", frameWindow.size);
	pt.put("frameWindow_step", frameWindow.step);
	pt.put("relativeSectionWindow_size", relativeSectionWindow.size);
	pt.put("relativeSectionWindow_step", relativeSectionWindow.step);
	pt.put("nOutputSections", nOutputSections);
	pt.put("flowSegmentationThreshold", flowSegmentationThreshold);
	pt.put("radius", radius);
	pt.put("cancellationDistance", cancellationDistance);
	return pt;
}

auto crowd::linecounting::LineLearningRepresenter::
create(boost::property_tree::ptree const& pt) -> std::unique_ptr<LineLearningRepresenter>
{
	return stdx::make_unique<LineLearningRepresenter>(
			*LineMultiFeatureExtractor::create(pt.get_child("extractor")),
			SlidingWindow<int>{pt.get<int>("frameWindow_size"), pt.get<int>("frameWindow_step")},
			SlidingWindow<double>{pt.get<double>("relativeSectionWindow_size"), pt.get<double>("relativeSectionWindow_step")},
			pt.get<int>("nOutputSections"),
			pt.get<double>("flowSegmentationThreshold"),
			pt.get<double>("radius"),
			pt.get<int>("cancellationDistance")

	);
}
