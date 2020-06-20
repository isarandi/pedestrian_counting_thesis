#include "cvret.hpp"

using namespace cv;

auto cvret::magnitude(InputArray xy) -> cv::Mat
{
    Mat m = xy.getMat();
    if (m.isContinuous())
    {
        m = m.reshape(1, xy.size().area());
        return cvret::magnitude(m.col(0), m.col(1)).reshape(1, xy.size().height);
    } else {
        return cvret::magnitude(cvret::extractChannel(xy, 0), cvret::extractChannel(xy,1));
    }
}

cv::Mat cvret::setTo(InputArray img, Scalar value, InputArray mask)
{
    Mat result = img.getMat().clone();
    result.setTo(value, mask);
    return result;
}
