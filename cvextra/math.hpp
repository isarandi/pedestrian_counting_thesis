#ifndef MATHUTILS_HPP
#define MATHUTILS_HPP

#include <opencv2/opencv.hpp>
#include "cvextra/core.hpp"
#include <vector>
#include <boost/numeric/interval.hpp>

namespace cvx {

namespace math
{

template <typename T>
auto clamp(T x, T minVal, T maxVal) -> T
{ return std::max(minVal, std::min(maxVal, x)); }

auto linearRescale(double x, double srcBegin, double srcEnd, double dstBegin, double dstEnd) -> double;

auto linspace(double startInclusive, double endInclusive, int n) -> std::vector<double>;
auto linspace2(double startInclusive, double endExclusive, int n) -> std::vector<double>;

auto percentileIndices(std::vector<double> vec, std::vector<double> percentiles) -> std::vector<int>;

auto bin(double val, double min, double max, int nBins) -> int;

auto standardNormalCdf(double x) -> double;

auto pearsonCorrelation(cv::InputArray src1, cv::InputArray src2) -> double;

}

auto rotationMatrix2D(double angle) -> cv::Matx22d;

inline
auto sq(double x) -> double {return x*x;}
inline
auto sq(int x) -> int {return x*x;}

inline
auto sq(cv::MatExpr const& x) -> cv::MatExpr {return x.mul(x);}

inline
auto sq(cv::Mat const& x) -> cv::MatExpr {return x.mul(x);}

inline
auto toRadians(double angleInDegrees) -> double
{
    return angleInDegrees*CV_PI/180.0;
}

inline
auto toDegrees(double angleInRadians) -> double
{
    return angleInRadians*180.0/CV_PI;
}



class LineSegmentProperties;

class LineSegment {
public:
    cv::Point2d p1;
    cv::Point2d p2;

    auto properties() const -> LineSegmentProperties;
    auto dir() const -> cv::Vec2d;
    auto clockwiseNormal() const -> cv::Vec2d;
    auto angleFromVerticalRadians() const -> double;
    auto length() const -> double;
    auto floorLength() const -> int;

    auto operator += (cv::Vec2d const& a) -> LineSegment&;
    auto operator -= (cv::Vec2d const& a) -> LineSegment&;
    auto operator *= (double a) -> LineSegment&;
    auto operator /= (double a) -> LineSegment&;

    struct IntersectionResult{
        bool intersects;
        cv::Point2d intersectionPoint;
    };

    static
	auto intersect(
            LineSegment const& seg1,
            LineSegment const& seg2
            ) -> IntersectionResult;

};

auto operator + (LineSegment const& ls1, cv::Vec2d const& a) -> LineSegment;
auto operator - (LineSegment const& ls1, cv::Vec2d const& a) -> LineSegment;
auto operator * (LineSegment const& ls1, double a) -> LineSegment;
auto operator / (LineSegment const& ls1, double a) -> LineSegment;

auto operator + (cv::Vec2d const& a, LineSegment const& ls1) -> LineSegment;
auto operator - (cv::Vec2d const& a, LineSegment const& ls1) -> LineSegment;
auto operator * (double a, LineSegment const& ls1) -> LineSegment;

class LineSegmentProperties {

public:
	LineSegmentProperties(LineSegment const& seg);

    auto localToGlobal(double pos) const -> cv::Point2d {return seg.p1+dir*pos;}
    auto globalToLocal(cv::Point2d const& pos) const -> double {return (pos-seg.p1).dot(dir);}

	LineSegment const seg;
	double const length;
	int const floorLength;
	cv::Vec2d const dir;
};

template <typename T>
class Range_
{
public:
    Range_(){}
    Range_(T start, T end)
        : start(start)
        , end(end){}

    T size() const {return end-start;}
    bool empty() const {return size==0;}

    static cvx::Range_<T> all()
    {
        return {std::numeric_limits<T>::min(),std::numeric_limits<T>::max()};
    }

    explicit operator cv::Range() const
    {
        return cv::Range{static_cast<int>(start), static_cast<int>(end)};
    }

    T start;
    T end;
};






}

#endif // MATHUTILS_HPP
