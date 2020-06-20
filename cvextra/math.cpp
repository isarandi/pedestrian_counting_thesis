#include "math.hpp"

#include <cvextra/core.hpp>
#include <cvextra/mats.hpp>
#include <cvextra/cvret.hpp>
#include <stdx/stdx.hpp>

#include <Eigen/Core>
#include <Eigen/Cholesky>
#include <cvextra/eigen.hpp>

#include <algorithm>
#include <utility>
#include <iostream>
#include <cmath>

using namespace std;
using namespace cvx;
using namespace cv;

auto cvx::math::
linspace(double a, double b, int n) -> vector<double>
{
    vector<double> v(n);
    double step = (b-a)/(n-1);

    for (int i = 0; i < n; ++i)
    {
        v[i] = a + i*step;
    }

    return v;
}

auto cvx::math::
linspace2(double a, double b, int n) -> std::vector<double>
{
    vector<double> v(n);
    double step = (b-a)/n;

    for (int i = 0; i < n; ++i)
    {
        v[i] = a + i*step;
    }

    return v;
}

auto cvx::math::
percentileIndices(
        vector<double> vec,
        vector<double> percentiles
        ) -> vector<int>
{
    vector<pair<int,double>> indexedVector;
    for (int i=0; i<vec.size(); ++i)
    {
        indexedVector.emplace_back(i, vec[i]);
    }

    auto compareSecond =
        [](pair<int,double> const& left, pair<int,double> const& right)
        {
            return left.second < right.second;
        };

    std::sort(indexedVector.begin(), indexedVector.end(), compareSecond);

    vector<int> result;
    for (auto& p : percentiles)
    {
        int idInSorted = static_cast<int>(std::round(p * (vec.size()-1)));
        int idInOriginal = indexedVector[idInSorted].first;
        result.push_back(idInOriginal);
    }

    return result;
}

auto cvx::math::
bin(
        double val,
        double min,
        double max,
        int nBins
        ) -> int
{
    return std::min(static_cast<int>(linearRescale(val, min, max, 0, nBins)), nBins-1);
}

auto cvx::math::
linearRescale(
        double x,
        double srcBegin,
        double srcEnd,
        double dstBegin,
        double dstEnd
        ) -> double
{
    if (srcEnd-srcBegin < std::numeric_limits<double>::epsilon())
    {
        return (dstBegin+dstEnd)*0.5;
    } else
    {
        return (x-srcBegin)/(srcEnd-srcBegin)*(dstEnd-dstBegin)+dstBegin;
    }
}

auto cvx::LineSegment::
intersect(
        const LineSegment &seg1,
        const LineSegment &seg2
        ) -> LineSegment::IntersectionResult
{
    double p0_x = seg1.p1.x;
    double p0_y = seg1.p1.y;
    double p1_x = seg1.p2.x;
    double p1_y = seg1.p2.y;
    double p2_x = seg2.p1.x;
    double p2_y = seg2.p1.y;
    double p3_x = seg2.p2.x;
    double p3_y = seg2.p2.y;

    double s1_x = p1_x - p0_x;
    double s1_y = p1_y - p0_y;
    double s2_x = p3_x - p2_x;
    double s2_y = p3_y - p2_y;

    double s = (-s1_y * (p0_x - p2_x) + s1_x * (p0_y - p2_y)) / (-s2_x * s1_y + s1_x * s2_y);
    double t = ( s2_x * (p0_y - p2_y) - s2_y * (p0_x - p2_x)) / (-s2_x * s1_y + s1_x * s2_y);

    double x = p0_x + (t * s1_x);
    double y = p0_y + (t * s1_y);

    if (s >= 0 && s <= 1 && t >= 0 && t <= 1)
    {
        return IntersectionResult{true, {x,y}};
    }

    return IntersectionResult{false, {x,y}};
}

auto cvx::LineSegment::
properties() const -> LineSegmentProperties
{
	return LineSegmentProperties(*this);
}

cvx::LineSegmentProperties::
LineSegmentProperties(LineSegment const& seg)
	: seg(seg)
	, length(cv::norm(seg.p2-seg.p1))
	, dir(length!=0 ? Vec2d(seg.p2-seg.p1)/length : Vec2d{0.,0.})
	, floorLength(static_cast<int>(length))
{
}

auto cvx::LineSegment::operator +=(const cv::Vec2d& a) -> LineSegment& {p1+=a;p2+=a; return *this;}
auto cvx::LineSegment::operator -=(const cv::Vec2d& a) -> LineSegment& {return *this+=-a;}
auto cvx::LineSegment::operator *=(double a) -> LineSegment& {p1*=a;p2*=a; return *this;}
auto cvx::LineSegment::operator /=(double a) -> LineSegment& {return *this*=1./a;}
auto cvx::operator +(const LineSegment& ls, const cv::Vec2d& a) -> LineSegment {LineSegment r=ls;return r+=a;}
auto cvx::operator -(const LineSegment& ls, const cv::Vec2d& a) -> LineSegment {return ls+(-a);}
auto cvx::operator *(const LineSegment& ls, double a) -> LineSegment {LineSegment r=ls;return r*=a;}
auto cvx::operator /(const LineSegment& ls, double a) -> LineSegment {return ls*(1./a);}
auto cvx::operator +(const cv::Vec2d& a, const LineSegment& ls) -> LineSegment {return ls+a;}
auto cvx::operator -(const cv::Vec2d& a, const LineSegment& ls) -> LineSegment {return ls-a;}
auto cvx::operator *(double a, const LineSegment& ls) -> LineSegment {
	return ls * a;
}

auto cvx::LineSegment::dir() const -> Vec2d {
	return (p1!=p2 ? Vec2d(p2-p1)/length() : Vec2d{0.,0.});
}
auto cvx::LineSegment::length() const -> double {
	return cv::norm(p2-p1);
}
auto cvx::LineSegment::floorLength() const -> int {
	return static_cast<int>(length());
}

auto cvx::LineSegment::
angleFromVerticalRadians() const -> double {
    auto d = dir();
    return std::atan2(d[1], d[0]) - CV_PI/2;
}

auto cvx::LineSegment::
clockwiseNormal() const -> cv::Vec2d
{
    auto d = dir();
    return Vec2d{-d[1], d[0]};
}

auto cvx::
rotationMatrix2D(double angle) -> Matx22d
{
    double c = std::cos(angle);
    double s = std::sin(angle);
    return Matx22d{c,s,-s,c};
}

auto cvx::math::
standardNormalCdf(double x) -> double
{
    static double minusOneOverSqrt2 = -1.0/std::sqrt(2);
    return 0.5 * std::erfc(x * minusOneOverSqrt2);
}

auto cvx::math::
pearsonCorrelation(InputArray src1, InputArray src2) -> double
{
    Mat reshaped1 = cvx::reshapeCols(src1,1,1);
    Mat reshaped2 = cvx::reshapeCols(src2,1,1);

    Mat1d covar;
    Mat mean;
    cv::calcCovarMatrix(cvret::hconcat(reshaped1, reshaped2), covar, mean, CV_COVAR_NORMAL | CV_COVAR_ROWS | CV_COVAR_SCALE);

    double pearsonCorrelation = covar(0,1) / std::sqrt((covar(0,0) * covar(1,1)));
    return pearsonCorrelation;
}
