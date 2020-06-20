#include "WindowBasedCounting.hpp"
#include <CrowdCounting/PersonLocations.hpp>
#include <CrowdCounting/LineCounting/FeatureSlices.hpp>
#include <CrowdCounting/LineCounting/SlidingWindow.hpp>
#include <CrowdCounting/RegionCounting/Features/Impl/features.hpp>
#include <MachineLearning/LearningSet.hpp>

#include <cvextra/coords.hpp>
#include <cvextra/core.hpp>
#include <cvextra/visualize.hpp>
#include <cvextra/vectors.hpp>
#include <cvextra/cvret.hpp>
#include <cvextra/cvenums.hpp>
#include <cvextra/improc.hpp>
#include <cvextra/LoopRange.hpp>
#include <cvextra/math.hpp>
#include <CrowdCounting/LineCounting/LineCounter.hpp>

#include <opencv2/opencv.hpp>

#include <cmath>
#include <limits>
#include <vector>

using namespace std;
using namespace cv;
using namespace cvx;
using namespace crowd;
using namespace crowd::linecounting;

auto sinc(double x) -> double
{
    if (std::abs(x) < std::numeric_limits<double>::epsilon())
    {
        return 1.0;
    } else {
        return std::sin(CV_PI*x)/(CV_PI*x);
    }
}

auto crowd::linecounting::
windowEstimatesToDeltas(
        Mat1d windowEstimates,
        SlidingWindow<int> slidingWindow,
        int nFrames
        ) -> Mat1d
{
    Mat1d deltas{Size{windowEstimates.cols, nFrames}, 0.0};

    //--- use the Whittaker–Shannon interpolation formula to reconstruct from the samples
    // i.e. convolution with sinc
    int centerOffset = (int)std::ceil(slidingWindow.size/2.0);
    int T = slidingWindow.step; // sampling period

    // the different columns correspond to independent window estimate sequences (e.g. the opposing directions)
    for (Point p : cvx::points(windowEstimates))
    {
        int centerPos = p.y*T + centerOffset;
        for (int t : cvx::irange(nFrames))
        {
            deltas(t, p.x) += windowEstimates(p) * sinc((double)(t - centerPos)/(double)T);
        }
    }

    // continue the values on both ends with repeating the end values and add their contributions
    // otherwise the sinc messes up the ends
    int nPhantomWindowEstimates = slidingWindow.size;
    for (int i : cvx::irange(nPhantomWindowEstimates))
    {
        int centerPos1 = -(i+1) * T + centerOffset;
        int centerPos2 = (windowEstimates.rows+i) * T + centerOffset;
        for (int t : cvx::irange(nFrames))
        {
            for (int j : cvx::irange(windowEstimates.cols))
            {
                deltas(t, j) += windowEstimates(0,j) * sinc((double)(t - centerPos1)/(double)T);
                deltas(t, j) += windowEstimates(windowEstimates.rows-1,j) * sinc((double)(t - centerPos2)/(double)T);
            }
        }
    }

    return deltas;
}

auto crowd::linecounting::
windowEstimatesToDeltasBoxFilter(
        Mat1d windowEstimates,
        SlidingWindow<int> slidingWindow,
        int nFrames
        ) -> Mat1d
{
    Mat1d deltas{Size{windowEstimates.cols, nFrames}, 0.0};

	double initValue = -1e8;
	deltas.setTo(initValue);

	// put the estimates to their centerpoint in the continuous solution
	int centerOffset = (int)std::ceil(slidingWindow.size/2.0);
	for (int iWindow : cvx::irange(windowEstimates.rows))
	{
		int centerPos = iWindow*slidingWindow.step+centerOffset;

		for (int iCol : cvx::irange(windowEstimates.cols))
		{
			deltas(centerPos,iCol) = windowEstimates(iWindow,iCol);
		}
	}

	// filter it with a box filter that gets smaller around the borders, basically
	Mat1d filteredDeltas{deltas.size()};
	for (int iFrame : cvx::irange(nFrames))
	{
		Range windowAroundFrame{iFrame-centerOffset, iFrame+(slidingWindow.size-centerOffset)+1};
		Range validWindow = cvx::intersect(windowAroundFrame, Range{0,nFrames});
		Mat1d validEstimates = deltas(validWindow,Range::all());

		if (cv::countNonZero(validEstimates!=initValue) == 0)
		{
		    if (windowEstimates.rows == 0)
		    {
		        throw "no window estimates";
		    }
		    windowEstimates.row(windowEstimates.rows-1).copyTo(filteredDeltas.row(iFrame));

		} else {

            for (int iCol : cvx::irange(windowEstimates.cols))
            {
                double avgEstimate = cv::mean(validEstimates.col(iCol), validEstimates.col(0)!=initValue)[0];
                filteredDeltas(iFrame,iCol) = avgEstimate;
            }
		}
	}

	return filteredDeltas;
}


