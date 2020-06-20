#include "illustLineFlow.hpp"

using namespace std;
using namespace cv;
using namespace cvx;
using namespace cvx::math;

auto cvx::illust::
inspectFlow(
        Mat2d const& flow,
        Mat3b const& im1,
        Mat3b const& im2,
        LineSegment const& segment,
		int factor,
		int step
        ) -> std::vector<Mat>
{
    Mat3b bigIm1 = cvret::resize(im1, im1.size()*factor, 0,0, INTER_NEAREST);
    Mat3b bigIm2 = cvret::resize(im2, im2.size()*factor, 0,0, INTER_NEAREST);

    LineSegmentProperties segprop = segment.properties();
    for (int i : cvx::irange(0,segprop.floorLength, step))
    {
        auto pos = (segprop.localToGlobal(i) + Point2d{.5,.5}) * factor;
        auto vec = flow(i) * factor*1.5;

        cvx::arrow(bigIm1, pos, vec, cvx::YELLOW, 1);
        cvx::arrow(bigIm2, pos, vec, cvx::YELLOW, 1);
    }

    return {bigIm1, bigIm2};
}



