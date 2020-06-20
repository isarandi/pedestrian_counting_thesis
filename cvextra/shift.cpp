#include "shift.hpp"

#include <limits>
#include <opencv2/imgproc/imgproc.hpp>

using namespace std;
using namespace cv;
using namespace cvx;

void cvx::shift(
        InputArray src_,
        OutputArray dst_,
        cv::Vec2d delta,
        int fill,
        cv::Scalar value)
{

    // error checking
    assert(std::abs(delta[0]) < src_.size().width && std::abs(delta[1]) < src_.size().height);

    // split the shift into integer and subpixel components
    cv::Point2i deltai(std::abs(delta[0]), std::abs(delta[1]));
    cv::Point2f deltasub(std::abs(delta[0] - deltai.x), std::abs(delta[1] - deltai.y));

    // INTEGER SHIFT
    // first create a border around the parts of the Mat that will be exposed
    int t = 0, b = 0, l = 0, r = 0;
    if (deltai.x > 0) l =  deltai.x;
    if (deltai.x < 0) r = -deltai.x;
    if (deltai.y > 0) t =  deltai.y;
    if (deltai.y < 0) b = -deltai.y;
    cv::Mat padded;
    cv::copyMakeBorder(src_, padded, t, b, l, r, fill, value);

    // SUBPIXEL SHIFT
    float eps = std::numeric_limits<float>::epsilon();
    if (deltasub.x > eps || deltasub.y > eps) {
        switch (src_.depth()) {
            case CV_32F:
            {
                cv::Matx<float, 1, 2> dx(1-deltasub.x, deltasub.x);
                cv::Matx<float, 2, 1> dy(1-deltasub.y, deltasub.y);
                sepFilter2D(padded, padded, -1, dx, dy, cv::Point(0,0), 0, cv::BORDER_CONSTANT);
                break;
            }
            case CV_64F:
            {
                cv::Matx<double, 1, 2> dx(1-deltasub.x, deltasub.x);
                cv::Matx<double, 2, 1> dy(1-deltasub.y, deltasub.y);
                sepFilter2D(padded, padded, -1, dx, dy, cv::Point(0,0), 0, cv::BORDER_CONSTANT);
                break;
            }
            default:
            {
                cv::Matx<float, 1, 2> dx(1-deltasub.x, deltasub.x);
                cv::Matx<float, 2, 1> dy(1-deltasub.y, deltasub.y);
                padded.convertTo(padded, CV_32F);
                sepFilter2D(padded, padded, CV_32F, dx, dy, cv::Point(0,0), 0, cv::BORDER_CONSTANT);
                break;
            }
        }
    }

    // construct the region of interest around the new matrix
    cv::Rect roi = cv::Rect(std::max(-deltai.x,0),std::max(-deltai.y,0),0,0) + src_.size();
    dst_.create(src_.size(), src_.type());
    cv::Mat dstMat = dst_.getMat();
    padded(roi).copyTo(dstMat);
}

auto cvxret::shift(
        cv::InputArray src,
        cv::Vec2d delta,
        int fill,
        cv::Scalar value
        ) -> cv::Mat
{
    cv::Mat result;
    cvx::shift(src, result, delta, fill, value);
    return result;
}

auto cvxret::shift(
        cv::InputArray src,
        cv::Vec2i delta,
        int fill,
        cv::Scalar value
        ) -> cv::Mat
{
    cv::Mat result;
    cvx::shift(src, result, delta, fill, value);
    return result;
}

void cvx::shift(
        cv::InputArray src,
        cv::OutputArray dst,
        cv::Vec2i delta,
        int fill,
        cv::Scalar value)
{
    cvx::shift(src,dst,cv::Vec2d{(double)delta[0],(double)delta[1]}, fill, value);
}
