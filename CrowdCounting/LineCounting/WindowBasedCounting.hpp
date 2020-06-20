#ifndef CROWDCOUNTING_LineCounting_WINDOWBASEDCOUNTING_HPP_
#define CROWDCOUNTING_LineCounting_WINDOWBASEDCOUNTING_HPP_

#include <CrowdCounting/LineCounting/SlidingWindow.hpp>
#include <CrowdCounting/LineCounting/FeatureSlices.hpp>
#include <MachineLearning/LearningSet.hpp>
#include <CrowdCounting/PersonLocations.hpp>
#include <opencv2/core/core.hpp>
#include <vector>

namespace crowd {


class LineCountingOptions
{
public:
    bool augmentWithMirrored;
    double kernelRidgeC;
    double kernelRidgeGamma;
    crowd::linecounting::SlidingWindow<double> relativeSectionWindow;
    crowd::linecounting::SlidingWindow<int> frameWindow;
    double signalVarianceMultiplier;
    double inputNoise;
};

namespace linecounting
{
/**
 * @brief getMovementMasks
 * @return Mat where 0 is no movement, 1 is leftward, 2 is rightward
 */
auto segmentMovement(
        cv::Mat2d const& flowSlice,
        cv::Mat3b const& imageSlice
        ) -> cv::Mat1b;

/**
 * Interpolates
 */
auto windowEstimatesToDeltas(
        cv::Mat1d windowEstimates,
        SlidingWindow<int> window,
        int nFrames
        ) -> cv::Mat1d;

auto windowEstimatesToDeltasBoxFilter(
        cv::Mat1d windowEstimates,
        SlidingWindow<int> window,
        int nFrames
        ) -> cv::Mat1d;

}

}


#endif /* CROWDCOUNTING_LineCounting_WINDOWBASEDCOUNTING_HPP_ */
