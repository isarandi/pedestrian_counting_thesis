#include <boost/range/irange.hpp>
#include <cvextra/coords.hpp>
#include <cvextra/core.hpp>
#include <cvextra/cvret.hpp>
#include <cvextra/LoopRange.hpp>
#include <cvextra/math.hpp>
#include <cvextra/mats.hpp>
#include <cvextra/vectors.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/core/mat.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/core/types_c.h>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include "localBinaryPatternsImpl.hpp"
#include <cassert>
#include <cmath>
#include <vector>

using namespace std;
using namespace cv;
using namespace cvx;
using namespace crowd;
using namespace crowd::localbinarypatterns;

vector<double> crowd::localBinaryPatternHistograms(
        InputArray input,
        double radius,
        int nSamples,
        int nHistogramBins,
        Size subGrid)
{
    Mat img = input.getMat();
    assert(nSamples < 31);

    vector<Vec2i> displacements;
    for (int iSample : cvx::irange(nSamples))
    {
        double angleRad = (2*CV_PI*iSample)/nSamples;
        Vec2i displacement = radius * Vec2d(std::cos(angleRad), std::sin(angleRad));
        displacements.push_back(displacement);
    }

    vector<double> result;
    result.reserve(nHistogramBins*subGrid.area());
    for (Point iSubgridCell : cvx::points(subGrid))
    {
        Rectd relativeRect = cvx::relativeGridRect(iSubgridCell, subGrid);
        Mat roi = cvx::extractRelativeRoi(img, relativeRect);

        vector<double> histogram = getHistogram(roi, displacements, nHistogramBins);
        cvx::vectors::push_back_all(result, histogram);
    }

    return result;
}

int crowd::localbinarypatterns::localDescriptor(
        InputArray src, Point location, vector<Vec2i> const& displacements)
{
    Mat1d img = src.getMat();
    Rect fullRect = cvx::fullRect(src);

    double referenceValue = img(location);

    int result = 0;
    for (auto& displacement : displacements)
    {
        result <<= 1;

        Point queryPoint = location + displacement;

        if (cvx::contains(fullRect, queryPoint))
        {
            double value = img(queryPoint);

            if (value > referenceValue)
            {
                result += 1;
            }
        }
    }

    return result;
}

int binaryToGrayCode(int num)
{
    return (num >> 1) ^ num;
}

void crowd::localbinarypatterns:: drawAll()
{
    auto examples = std::vector<Mat1b>(256);

    for (int i : cvx::irange(256))
    {
        int grayCode = binaryToGrayCode(i);

        auto mat = Mat1b(3,3);

        mat(0,0) = (i&1)*255;
        i >>= 1;
        mat(0,1) = (i&1)*255;
        i >>= 1;
        mat(0,2) = (i&1)*255;
        i >>= 1;
        mat(1,2) = (i&1)*255;
        i >>= 1;
        mat(2,2) = (i&1)*255;
        i >>= 1;
        mat(2,1) = (i&1)*255;
        i >>= 1;
        mat(2,0) = (i&1)*255;
        i >>= 1;
        mat(1,0) = (i&1)*255;

        mat(1,1) = 128;
        examples[grayCode] = mat;
    }

    for (int i: cvx::irange(256))
    {
        cv::imshow("LBP", cvret::resize(examples[i], Size(300,300), 0, 0, INTER_NEAREST));
        cv::waitKey();
    }
}

vector<double> crowd::localbinarypatterns::getHistogram(
        InputArray input,
        vector<Vec2i> const& diffs,
        int nHistogramBins)
{

    Mat img = input.getMat();
    vector<double> histogram(nHistogramBins, 0.0);

    int maxLocalValue = (1 << diffs.size())-1;

    for (Point const& p : cvx::points(img))
    {
        int value = localDescriptor(img, p, diffs);

        if (value == 0)
        {
            ++histogram[0];
        } else {
            int grayCode = binaryToGrayCode(value);
            int bin = cvx::math::bin(grayCode, 1, maxLocalValue, nHistogramBins-1) + 1;
            ++histogram[bin];
        }
    }

    return histogram;
}

