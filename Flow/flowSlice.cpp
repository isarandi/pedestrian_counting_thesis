#include "flowSlice.hpp"
#include "lineFlow.hpp"
#include "featuresForFlow.hpp"
#include "Illustrate/illustrate.hpp"
#include <cvextra.hpp>
#include <iostream>
#include <string>
#include <queue>
#include <omp.h>

using namespace std;
using namespace cv;
using namespace cvx;
using namespace crowd::lineopticalflow;
using namespace crowd::lineopticalflow::details;

auto guiGetLineSegment(Mat image) -> LineSegment;

auto weightSum(int T, double beta) -> double
{
    double sum =0;
    for (int i : cvx::irange(-T,T))
    {
        if (i!=0)
        {
            sum += std::exp(-beta*std::abs(i));
        }
    }
    return sum;
}

template <typename ImageSequence>
auto crowd::lineopticalflow::
createFlowSlice(
		ImageSequence images,
        LineSegment const& segment,
        int T,
        int indexOffset,
        OpticalFlowOptions const& options
        ) -> Mat2d
{
    auto calculateFeaturesForFlow = crowd::lineopticalflow::details::calcRGBAndGrad;

    //--- Set up the coordinate system C where the line is vertical
    double segmentAngleFromVerticalRadians = segment.angleFromVerticalRadians();

    int roiCoreWingSize = (1<<(options.nScales-1));
    int pyramidKernelWingSize = 2;
    int roiWingSize = roiCoreWingSize * (pyramidKernelWingSize+1);
    int roiWidth = roiWingSize*2+1;

    int nLineSamples = segment.floorLength();
    Size roiSize{roiWidth, nLineSamples};

    //--- Rotated rect around line in default coord.sys.
    RotatedRect roiRect(
                cvx::point2f(segment.p1+segment.p2)/2,
                roiSize,
                cvx::toDegrees(segmentAngleFromVerticalRadians));
    Mat extractorMatrix = getRotatedRectExtractorMatrix(roiRect); // warps to coord.sys. C

    //---
    list<Mat> imageQueue;
    int queueSize = 2;//*T+1;
    auto imIter = images.begin();
    vector<Mat> colorImages;

    //--- Fill the frame queue to be able to start
    for (int iElement : cvx::irange(queueSize))
    {
        Mat3b roiPart = cvret::warpAffine(*(imIter++), extractorMatrix, roiSize);
        colorImages.push_back(roiPart);
        Mat convertedMat = calculateFeaturesForFlow(roiPart);
        cv::GaussianBlur(convertedMat, convertedMat, {9,9}, 3);
        imageQueue.push_back(convertedMat);
    }
    //---

    Mat2d flowSlice{0, nLineSamples};

    vector<double> channelWeights = {1/5.,1/5.,1/5.,1/5.,1/5.};

    for (int iFrame = queueSize-1; ; ++iFrame)
    {
        cvx::io::statusUpdate(cvx::str::format("Processing flow for frame #%d", iFrame+indexOffset));

        Mat2d flow = crowd::lineopticalflow::solveOpticalFlow(
                    std::vector<Mat>(imageQueue.begin(), imageQueue.end()),
                    channelWeights,
                    0,
                    Mat2d::zeros(1,nLineSamples),
                    options);
        cvx::vconcat(flowSlice, flow, flowSlice);

        //--- Step the FIFO queue
        imageQueue.pop_front();

        if (imIter == images.end())
        {
            break;
        }

        //--- Take next
        Mat3b roiPart = cvret::warpAffine(*(imIter++), extractorMatrix, roiSize);
        colorImages.push_back(roiPart);
        Mat convertedMat = calculateFeaturesForFlow(roiPart);
        cv::GaussianBlur(convertedMat, convertedMat, {9,9}, 3);
        imageQueue.push_back(convertedMat);
        //---
    }

    //--- Rotate the flows to the original orientation
    Matx22d rotatorMatrix = cvx::rotationMatrix2D(segmentAngleFromVerticalRadians);

    for (Point p : cvx::points(flowSlice))
    {
        flowSlice(p) = cvx::vec(rotatorMatrix * flowSlice(p));
    }

    return flowSlice;
}

auto crowd::lineopticalflow::
createFlowSliceParallel(
        ImageLoadingIterable const& images,
        LineSegment const& segment,
        OpticalFlowOptions const& options
        ) -> Mat2d
{
	//--- Do the flow computations in parallel
	int nImages = images.size();
    int nParallelBlocks = 8;
    int blockSize = (nImages/nParallelBlocks);

    Mat2d flowSlice{nImages, segment.floorLength()};

    #pragma omp parallel for
    for (int iBlock=0; iBlock<nParallelBlocks; ++iBlock)
    {
        int iStart = iBlock * blockSize;
        int iEndFlow  = (iBlock < nParallelBlocks-1) ? (iBlock+1) * blockSize+1 : nImages;

        Mat flowPart =
                createFlowSlice(
                    images.range(iStart, iEndFlow),
                    segment,
                    0,
                    iStart,
                    options);

        Mat flowTarget = flowSlice.rowRange(iStart,iEndFlow-1);
        flowPart.copyTo(flowTarget);
    }

    return flowSlice;
}

// instantiate
#define INSTANTIATE_CREATE_FLOW_SLICE(T) \
template \
auto crowd::lineopticalflow::createFlowSlice<T >( \
		T images, \
        LineSegment const& segment, \
        int t, \
        int indexOffset, \
        OpticalFlowOptions const& options \
        ) -> Mat2d;


INSTANTIATE_CREATE_FLOW_SLICE(cvx::ImageLoadingIterable)
INSTANTIATE_CREATE_FLOW_SLICE(std::vector<Mat>)
INSTANTIATE_CREATE_FLOW_SLICE(cvx::Iterable<cvx::ImageLoadingIterable::iterator_t>)
