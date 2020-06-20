#ifndef CROWDCOUNTING_LineCounting_LINELEARNINGREPRESENTER_HPP_
#define CROWDCOUNTING_LineCounting_LINELEARNINGREPRESENTER_HPP_

#include <CrowdCounting/LineCounting/Features/LineMultiFeatureExtractor.hpp>
#include <CrowdCounting/LineCounting/SlidingWindow.hpp>
#include <CrowdCounting/LineCounting/FeatureSlices.hpp>
#include <MachineLearning/Regression.hpp>
#include <CrowdCounting/PersonLocations.hpp>
#include <opencv2/core/core.hpp>

namespace crowd {
class LearningSet;
class LineCountingOptions;
} /* namespace crowd */

namespace crowd {
namespace linecounting {

class LineLearningRepresenter
{
public:
	LineLearningRepresenter(
			LineMultiFeatureExtractor const& extractor,
			SlidingWindow<int> frameWindow,
			SlidingWindow<double> relativeSectionWindow,
			int nOutputSections,
			double flowSegmentationThreshold,
			double radius=1e-3,
			int cancellationDistance = 0);

    LineLearningRepresenter(
            LineMultiFeatureExtractor const& extractor,
            SlidingWindow<int> frameWindow,
            int nMicroSections,
            int sectionSizeInMicroSections,
            int sectionStepInMicroSections,
            int nOutputSections,
            double flowSegmentationThreshold,
            double radius=1e-3,
            int cancellationDistance = 0);

    auto toLearningSet(
    		FeatureSlices slices,
			cvx::LineSegment const& targetSegment,
			PersonLocations const& locations
    		) const -> LearningSet;

    auto toContinuousSolution(
    		PredictionWithConfidence const& pc,
    		int nFrames
			) const-> PredictionWithConfidence;

    auto createTimeWindowOutputs(
            PersonLocations const& locations,
            cvx::LineSegment const& targetSegment,
			int nFrames
            ) const -> cv::Mat1d;

    auto createTimeWindowDescriptors(
            FeatureSlices const& slices,
            cvx::LineSegment const& targetSegment
            ) const -> cv::Mat1d;

    auto createSingleSectionWindowDescriptors(
            FeatureSlices const& slices
            ) const -> cv::Mat1d;

    auto segmentMovement(
            cv::Mat2d const& flowSlice,
			cv::Mat3b const& imageSlice
			) const -> cv::Mat1b;

    static
	auto segmentMovementStatic(
            cv::Mat2d const& flowSlice,
			cv::Mat3b const& imageSlice,
			double threshold
			) -> cv::Mat1b;

    CVX_CONFIG_DERIVED(LineLearningRepresenter)

    virtual ~LineLearningRepresenter(){}

private:
    LineMultiFeatureExtractor extractor;
    SlidingWindow<int> frameWindow;
    SlidingWindow<double> relativeSectionWindow;

    int nOutputSections;
    double radius;

    double flowSegmentationThreshold;
    int cancellationDistance;
};

} /* namespace linecounting */
} /* namespace crowd */

#endif /* CROWDCOUNTING_LineCounting_LINELEARNINGREPRESENTER_HPP_ */
