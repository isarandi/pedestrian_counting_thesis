#include "colors.hpp"
#include "mats.hpp"
#include "cvret.hpp"

using namespace std;
using namespace cv;
using namespace cvx;

auto cvx::cvtColor(Vec3b const& color, int conversionType) -> Vec3b
{
    Mat3b m{1,1};
    m(0,0) = color;
    Mat3b res = cvret::cvtColor(m, conversionType);
    return res(0,0);
}

auto cvx::cvtColor(Scalar const& color, int conversionType) -> Scalar
{
    Mat4b m;
    m(0,0) = color;
    Mat4d result = cvret::cvtColor(m, conversionType);
    Vec4d r = result(0,0);

    return {r[0],r[1],r[2],r[3]};
}

auto cvx::toVec3b(cv::Scalar const& color) -> cv::Vec3b
{
    return {(uchar)color[0], (uchar)color[1], (uchar)color[2]};
}
auto cvx::toScalar(cv::Vec3b const& color) -> cv::Scalar
{
    return {(double)color[0], (double)color[1], (double)color[2]};
}
