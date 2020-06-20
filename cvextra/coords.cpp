#include "coords.hpp"
#include <opencv2/imgproc/imgproc.hpp>

using namespace std;
using namespace cv;
using namespace cvx;


Point cvx::center(InputArray input)
{
    return cvx::center(input.size());
}

Rect cvx::fullRect(InputArray input)
{
    return cvx::fullRect(input.size());
}

Point cvx::borderInterpolate(Point p, Size size, int borderType)
{
    return cvx::borderInterpolate(p, size, borderType, borderType);
}

Point cvx::borderInterpolate(Point p, Size size, int borderTypeX, int borderTypeY)
{
    return Point(
                cv::borderInterpolate(p.x, size.width, borderTypeX),
                cv::borderInterpolate(p.y, size.height, borderTypeY));
}


Rectd cvx::relativeGridRect(Point gridIndices, Size gridResolution)
{
    return Rectd(
                gridIndices.x/(double)gridResolution.width,
                gridIndices.y/(double)gridResolution.height,
                1.0/gridResolution.width,
                1.0/gridResolution.height);

}

bool cvx::contains(Range const& range, int number)
{
    return range.start <= number && number < range.end;
}

bool cvx::contains(Range const& r1, Range const& r2)
{
    return r1.start <= r2.start && r2.end <= r1.end;
}

Rect cvx::intersect(Rect r1, Rect r2)
{
    int xStart = std::max(r1.x, r2.x);
    int yStart = std::max(r1.y, r2.y);

    int xEnd = std::min(r1.x+r1.width, r2.x+r2.width);
    int yEnd = std::min(r1.y+r1.height, r2.y+r2.height);

    int xSize = std::max(0, xEnd-xStart);
    int ySize = std::max(0, yEnd-yStart);

    return Rect{xStart, yStart, xSize, ySize};
}

auto cvx::intersect(
        cv::Range r1,
        cv::Range r2
        ) -> cv::Range
{
    int start = std::max(r1.start, r2.start);
    int end =  std::min(r1.end, r2.end);

    if (end < start)
    {
        end = start;
    }

    return Range{start, end};
}

Point cvx::rel2abs(Point2d relPoint, Size size)
{
    return Point(
                static_cast<int>(relPoint.x*size.width),
                static_cast<int>(relPoint.y*size.height));
}

Point2d cvx::abs2rel(Point absPoint, Size size)
{
    return Point2d(absPoint.x*size.width, absPoint.y*size.height);
}

Rect cvx::rel2abs(Rectd relativeRoi, Size imageSize)
{
    int x = std::max(0, (int)(std::ceil(relativeRoi.x*imageSize.width)));
    int y = std::max(0, (int)(std::ceil(relativeRoi.y*imageSize.height)));

    int width = std::min(
                imageSize.width-x,
                (int)(std::ceil(relativeRoi.width * imageSize.width)));

    int height = std::min(
                imageSize.height-y,
                (int)(std::ceil(relativeRoi.height * imageSize.height)));

    return Rect(x,y,width,height);
}

Rectd cvx::abs2rel(Rect absRect, Size size)
{
    return Rectd(
                static_cast<double>(absRect.x) / size.width,
                static_cast<double>(absRect.y) / size.height,
                static_cast<double>(absRect.width) / size.width,
                static_cast<double>(absRect.height) / size.height);
}

