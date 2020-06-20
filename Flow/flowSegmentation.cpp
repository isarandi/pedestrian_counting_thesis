//extern "C" {
//#include <vl/slic.h>
//}

#include "lineFlow.hpp"
#include <cvextra.hpp>
#include <iostream>
#include <string>

using namespace std;
using namespace cv;
using namespace cvx;
using namespace cvx::math;

//void slicSegment(InputArray imageData,
//                 OutputArray labels,
//                 int regionSize,
//                 double regularization,
//                 double minRegionSize)
//{
//    Mat imageMat = cvret::convertType(imageData.getMat(), CV_32F);
//
//    labels.create(imageMat.size(), CV_32S);
//    Mat1i labelsMat = labels.getMat();
//
//    vl_slic_segment(
//                labelsMat.ptr<vl_uint32>(),
//                imageMat.ptr<float>(),
//                imageMat.cols,
//                imageMat.rows,
//                imageMat.channels(),
//                regionSize,
//                regularization,
//                minRegionSize);
//}
//
//auto slicSegment(
//        InputArray imageData,
//        int regionSize,
//        double regularization,
//        double minRegionSize) -> Mat1i
//{
//    Mat1i result;
//    slicSegment(imageData, result, regionSize, regularization, minRegionSize);
//    return result;
//}

auto getBorders(Mat1i labels) -> BinaryMat
{
    BinaryMat borders = Mat1b::zeros(labels.size());

    double min;
    double max;
    cv::minMaxLoc(labels, &min, &max);

    for (int label : cvx::irange(static_cast<int>(min), static_cast<int>(max)))
    {
        cv::max(borders, cvxret::outline(labels == label), borders);
    }

    Mat structElem = cv::getStructuringElement(MORPH_ELLIPSE, {3,3});
    cv::erode(borders, borders, structElem);

    return borders;
}

void testSegmentation()
{

}
