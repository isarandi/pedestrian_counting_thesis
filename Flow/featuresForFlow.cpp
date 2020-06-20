#include "featuresForFlow.hpp"
#include <cvextra.hpp>

using namespace std;
using namespace cv;
using namespace cvx;
using namespace cvx::math;

using namespace crowd;

auto crowd::lineopticalflow::details::
calcCartesianHueSat(Mat3b const& bgrImage) -> Mat2d
{
    Mat3d hsvImage = cvret::cvtColor(bgrImage, cv::COLOR_BGR2HSV);
    Mat1d x;
    Mat1d y;

    cv::polarToCart(
                cvret::extractChannel(hsvImage, 1)/255,
                cvret::extractChannel(hsvImage, 0)/180,
                x, y);

    return cvret::merge({x,y});
}

auto crowd::lineopticalflow::details::
calcHueSat(Mat3b const& bgrImage) -> Mat2d
{
    Mat3d hsvImage = cvret::cvtColor(bgrImage, cv::COLOR_BGR2HSV);
    return cvret::merge({cvret::extractChannel(hsvImage, 1)/255,cvret::extractChannel(hsvImage, 0)/180});
}

auto crowd::lineopticalflow::details::
calcNormRGB(Mat3b const& bgrImage) -> Mat3d
{
    Mat3d doubleBgr = Mat3d{bgrImage}/255;

    Mat1d b = cvret::extractChannel(doubleBgr, 0);
    Mat1d g = cvret::extractChannel(doubleBgr, 1);
    Mat1d r = cvret::extractChannel(doubleBgr, 2);

    Mat1d sum = b+g+r;

    return cvret::merge({cvret::divide(b,sum),cvret::divide(g,sum), cvret::divide(r,sum)})*3;
}

auto crowd::lineopticalflow::details::
calcRGB(Mat3b const& bgrImage) -> Mat3d
{
    Mat3d doubleBgr = Mat3d{bgrImage}/255;
    return doubleBgr;
}

auto crowd::lineopticalflow::details::
calcRGBAndGrad(Mat3b const& bgrImage) -> Mat
{
    Mat3d doubleBgr = Mat3d{bgrImage}/255;
    Mat1d intensity = Mat1d{cvret::cvtColor(bgrImage, cv::COLOR_BGR2GRAY)}/255;
    Mat2d gradx = (cvret::Sobel(intensity, CV_64F, 1, 0)+8)/16;
    Mat2d grady = (cvret::Sobel(intensity, CV_64F, 0, 1)+8)/16;

    return cvret::merge({doubleBgr,
                         gradx,
                         grady});
}

auto crowd::lineopticalflow::details::
calcCIEab(Mat3b const& bgrImage) -> Mat2d
{
    Mat3d lab;
    cv::cvtColor(bgrImage, lab, cv::COLOR_BGR2Lab);

    return (cvret::merge({cvret::extractChannel(lab, 1),cvret::extractChannel(lab, 0)})+127)/255;
}

auto crowd::lineopticalflow::details::
calcGray(Mat3b const& bgrImage) -> Mat1d
{
    Mat1d gray = cvret::cvtColor(bgrImage, cv::COLOR_BGR2GRAY);
    return gray/255.;
}
