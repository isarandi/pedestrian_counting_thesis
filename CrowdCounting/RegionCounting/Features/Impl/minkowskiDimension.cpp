#include "features.hpp"
#include <cvextra/core.hpp>
#include <cvextra/cvret.hpp>
#include <cvextra/LoopRange.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/core/mat.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <cmath>

using namespace std;
using namespace cv;
using namespace cvx;

double crowd::Minkowski(InputArray src, int maxRadius)
{
    if (cv::countNonZero(src) == 0)
    {
        return 0;
    }

    int radiusStep = 1;
    int nDiskSteps = maxRadius / radiusStep;

    BinaryMat img = src.getMat();
    BinaryMat dilated{img.size()};

    Mat1d X{nDiskSteps, 2};
    Mat1d Y{nDiskSteps, 1};

    X(0,0) = 1.0;
    X(0,1) = 0.0;
    Y(0,0) = std::log(cv::countNonZero(img));

    int row = 1;
    for (int diskRadius : cvx::irange(radiusStep, maxRadius, radiusStep))
    {
        int diskSize = diskRadius*2 + 1;
        cv::dilate(img, dilated, cv::getStructuringElement(MORPH_ELLIPSE, Size{diskSize,diskSize}));

        X(row,0) = 1.0;
        X(row,1) = std::log(diskSize);

        Y(row,0) = std::log(cv::countNonZero(dilated));
        ++row;
    }

    Mat1d W = cvret::solve(X, Y, DECOMP_QR);
    double slope = W(1, 0);

    return 2 - slope;
}
