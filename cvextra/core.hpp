#ifndef OPERATORS_HPP
#define OPERATORS_HPP

#include <opencv2/core/core.hpp>

/**
 * Additional useful typedef and operators that are missing from standard OpenCV.
 */
namespace cvx {

enum {SRC_TYPE = -1};

typedef cv::Vec<float,  1> Vec1f;
typedef cv::Vec<float,  5> Vec5f;
typedef cv::Vec<double, 1> Vec1d;
typedef cv::Vec<double, 5> Vec5d;
typedef cv::Mat_<cv::Vec<float,  5>> Mat5f;
typedef cv::Mat_<cv::Vec<double, 5>> Mat5d;
typedef cv::Rect_<double> Rectd;
typedef cv::Mat1b BinaryMat;
typedef cv::Size_<double> Size2d;

template <typename _Tp>
auto point2f(cv::Point_<_Tp> const& p) -> cv::Point2f
{
    return {static_cast<float>(p.x), static_cast<float>(p.y)};
}

template <typename _Tp>
auto point2d(cv::Point_<_Tp> const& p) -> cv::Point2d
{
    return {static_cast<double>(p.x), static_cast<double>(p.y)};
}

template <typename _Tp>
auto point2i(cv::Point_<_Tp> const& p) -> cv::Point
{
    return {static_cast<int>(p.x), static_cast<int>(p.y)};
}

template <typename _Tp>
auto size2f(cv::Size_<_Tp> const& s) -> cv::Size2f
{
    return {static_cast<float>(s.width), static_cast<float>(s.height)};
}

template <typename _Tp>
auto size2d(cv::Size_<_Tp> const& s) -> Size2d
{
    return {static_cast<double>(s.width), static_cast<double>(s.height)};
}

template <typename _Tp>
auto size2i(cv::Size_<_Tp> const& s) -> cv::Size
{
    return {static_cast<int>(s.width), static_cast<int>(s.height)};
}

template <typename _Tp, int Dim>
auto vec(cv::Matx<_Tp, Dim, 1>& matx) -> cv::Vec<_Tp, Dim>&
{
    return reinterpret_cast<cv::Vec<_Tp, Dim>&>(matx);
}

template <typename _Tp, int Dim>
auto vec(cv::Matx<_Tp, Dim, 1> const& matx) -> cv::Vec<_Tp, Dim> const&
{
    return reinterpret_cast<cv::Vec<_Tp, Dim> const&>(matx);
}

//::
// Operators for Point, so that it's easy to combine them with Vec objects.
// Point should stand for a position, while Vec for a displacement (there may be borderline cases).
//::
template <typename _Tp1, typename _Tp2> inline
auto operator +(
        cv::Point_<_Tp1> const& point,
        cv::Vec<_Tp2,2> const& vec
        ) -> cv::Point_<typename std::common_type<_Tp1,_Tp2>::type>
{
    return {point.x+vec[0], point.y+vec[1]};
}

template <typename _Tp1, typename _Tp2> inline
auto operator -(
        cv::Point_<_Tp1> const& point,
        cv::Vec<_Tp2,2> const& vec
        ) -> cv::Point_<typename std::common_type<_Tp1,_Tp2>::type>
{
    return {point.x-vec[0], point.y-vec[1]};
}

template <typename _Tp1, typename _Tp2> inline
auto operator +(
        cv::Point_<_Tp1> const& point,
        cv::Size_<_Tp2> const& size
        ) -> cv::Point_<typename std::common_type<_Tp1,_Tp2>::type>
{
    return {point.x+size.width, point.y+size.height};
}

template <typename _Tp1, typename _Tp2> inline
auto operator -(
        cv::Point_<_Tp1> const& point,
        cv::Size_<_Tp2> const& size
        ) -> cv::Point_<typename std::common_type<_Tp1,_Tp2>::type>
{
    return {point.x-size.width, point.y-size.height};
}

template <typename _Tp> inline
auto operator += (cv::Point_<_Tp>& point, cv::Vec<_Tp,2> const& vec) -> cv::Point_<_Tp>&
{
    return point += static_cast<cv::Point_<_Tp> >(vec);
}

template <typename _Tp> inline
auto operator -= (cv::Point_<_Tp>& point, cv::Vec<_Tp,2> const& vec) -> cv::Point_<_Tp>&
{
    return point -= static_cast<cv::Point_<_Tp> >(vec);
}

template <typename _Tp1, typename _Tp2> inline
auto operator + (
        cv::Point3_<_Tp1> const& point,
        cv::Vec<_Tp2,3> const& vec
        ) -> cv::Point3_<typename std::common_type<_Tp1,_Tp2>::type>
{
    return {point.x+vec[0], point.y+vec[1], point.z+vec[2]};
}

template <typename _Tp1, typename _Tp2> inline
auto operator - (
        cv::Point3_<_Tp1> const& point,
        cv::Vec<_Tp2,3> const& vec
        ) -> cv::Point3_<typename std::common_type<_Tp1,_Tp2>::type>
{
    return {point.x-vec[0], point.y-vec[1], point.z-vec[2]};
}

template <typename _Tp> inline
auto operator += (cv::Point3_<_Tp>& point, cv::Vec<_Tp,3> const& vec) -> cv::Point3_<_Tp>&
{
    return point += static_cast<cv::Point3_<_Tp> >(vec);
}

template <typename _Tp> inline
auto operator -= (cv::Point3_<_Tp>& point, cv::Vec<_Tp,3> const& vec) -> cv::Point3_<_Tp>&
{
    return point -= static_cast<cv::Point3_<_Tp> >(vec);
}

//::
// Division and multiplication operators
//::

//---
// Point_

template<typename _Tp, typename _Tp2> inline
auto operator / (cv::Point_<_Tp> const& a, _Tp2 alpha) -> cv::Point_<_Tp>
{
    return {a.x/alpha, a.y/alpha};
}

template<typename _Tp, typename _Tp2> inline
auto operator /= (cv::Point_<_Tp>& a, _Tp2 alpha) -> cv::Point_<_Tp>&
{
    a.x /= alpha;
    a.y /= alpha;
    return a;
}

//---
// Point_ with Size_

template<typename _Tp1, typename _Tp2> inline
auto operator / (cv::Point_<_Tp1> const& a, cv::Size_<_Tp2> const& b) -> cv::Point_<typename std::common_type<_Tp1,_Tp2>::type>
{
	typedef typename std::common_type<_Tp1,_Tp2>::type _Tp;
    return {static_cast<_Tp>(a.x)/static_cast<_Tp>(b.width), static_cast<_Tp>(a.y)/static_cast<_Tp>(b.height)};
}

template<typename _Tp, typename _Tp2> inline
auto operator /= (cv::Point_<_Tp>& a, cv::Size_<_Tp2> const& b) -> cv::Point_<_Tp>&
{
    a.x /= b.width;
    a.y /= b.height;
    return a;
}

template<typename _Tp1, typename _Tp2> inline
auto operator * (cv::Point_<_Tp1> const& a, cv::Size_<_Tp2> const& b) -> cv::Point_<typename std::common_type<_Tp1,_Tp2>::type>
{
	typedef typename std::common_type<_Tp1,_Tp2>::type _Tp;
    return {static_cast<_Tp>(a.x)*static_cast<_Tp>(b.width), static_cast<_Tp>(a.y)*static_cast<_Tp>(b.height)};
}

template<typename _Tp, typename _Tp2> inline
auto operator *= (cv::Point_<_Tp>& a, cv::Size_<_Tp2> const& b) -> cv::Point_<_Tp>&
{
    a.x *= b.width;
    a.y *= b.height;
    return a;
}

template<typename _Tp1, typename _Tp2> inline
auto operator * (cv::Size_<_Tp2> const& b, cv::Point_<_Tp1> const& a) -> cv::Point_<typename std::common_type<_Tp1,_Tp2>::type>
{
	return a*b;
}

//---
// Vec_ with Size_

template<typename _Tp1, typename _Tp2> inline
auto operator / (cv::Vec<_Tp1,2> const& a, cv::Size_<_Tp2> const& b) -> cv::Vec<typename std::common_type<_Tp1,_Tp2>::type,2>
{
    typedef typename std::common_type<_Tp1,_Tp2>::type _Tp;
    return {static_cast<_Tp>(a[0])/static_cast<_Tp>(b.width), static_cast<_Tp>(a[1])/static_cast<_Tp>(b.height)};
}

template<typename _Tp, typename _Tp2> inline
auto operator /= (cv::Vec<_Tp,2>& a, cv::Size_<_Tp2> const& b) -> cv::Vec<_Tp,2>&
{
    a[0] /= b.width;
    a[1] /= b.height;
    return a;
}

template<typename _Tp1, typename _Tp2> inline
auto operator * (cv::Vec<_Tp1,2> const& a, cv::Size_<_Tp2> const& b) -> cv::Vec<typename std::common_type<_Tp1,_Tp2>::type,2>
{
    typedef typename std::common_type<_Tp1,_Tp2>::type _Tp;
    return {static_cast<_Tp>(a[0])*static_cast<_Tp>(b.width), static_cast<_Tp>(a[1])*static_cast<_Tp>(b.height)};
}

template<typename _Tp, typename _Tp2> inline
auto operator *= (cv::Vec<_Tp,2>& a, cv::Size_<_Tp2> const& b) -> cv::Vec<_Tp,2>&
{
    a[0] *= b.width;
    a[1] *= b.height;
    return a;
}

template<typename _Tp1, typename _Tp2> inline
auto operator * (cv::Size_<_Tp2> const& b, cv::Vec<_Tp1,2> const& a) -> cv::Vec<typename std::common_type<_Tp1,_Tp2>::type,2>
{
    return a*b;
}

//---
// Point3_

template<typename _Tp, typename _Tp2> inline
auto operator / (cv::Point3_<_Tp> const& a, _Tp2 alpha) -> cv::Point3_<_Tp>
{
    return {a.x*alpha, a.y*alpha, a.z*alpha};
}

template<typename _Tp, typename _Tp2> inline
auto operator /= (cv::Point3_<_Tp>& a, _Tp2 alpha) -> cv::Point3_<_Tp>&
{
    a.x /= alpha;
    a.z /= alpha;
    a.y /= alpha;
    return a;
}

//---
// Vec
template<typename _Tp, typename _Tp2, int cn> inline
auto operator / (cv::Vec<_Tp, cn> const& a, _Tp2 alpha) -> cv::Vec<_Tp, cn>
{
    cv::Vec<_Tp, cn> result;
    for (int i=0; i<cn; ++i)
        result[i] = a[i]/alpha;
    return result;
}

template<typename _Tp, typename _Tp2, int cn> inline
auto operator /= (cv::Vec<_Tp, cn>& a, _Tp2 alpha) -> cv::Vec<_Tp, cn>&
{
    for (int i=0; i<cn; ++i)
        a[i] /= alpha;
    return a;
}

//---
// Size

template<typename _Tp, typename _Tp2> inline
auto operator / (cv::Size_<_Tp> const& a, _Tp2 alpha) -> cv::Size_<_Tp>
{
    return {a.width/alpha, a.height/alpha};
}

template<typename _Tp, typename _Tp2> inline
auto operator /= (cv::Size_<_Tp>& a, _Tp2 alpha) -> cv::Size_<_Tp>&
{
    a.width /= alpha;
    a.height /= alpha;
    return a;
}

template<typename _Tp, typename _Tp2> inline
auto operator * (cv::Size_<_Tp> const& a, _Tp2 alpha) -> cv::Size_<_Tp>
{
    return {a.width*alpha, a.height*alpha};
}

template<typename _Tp, typename _Tp2> inline
auto operator *= (cv::Size_<_Tp>& a, _Tp2 alpha) -> cv::Size_<_Tp>&
{
    a.width *= alpha;
    a.height *= alpha;
    return a;
}

//---
// Size with Size

template<typename _Tp1, typename _Tp2> inline
auto operator / (cv::Size_<_Tp1> const& a, cv::Size_<_Tp2> const& b) -> cv::Size_<typename std::common_type<_Tp1,_Tp2>::type>
{
	typedef typename std::common_type<_Tp1,_Tp2>::type _Tp;
    return {static_cast<_Tp>(a.width)/static_cast<_Tp>(b.width), static_cast<_Tp>(a.height)/static_cast<_Tp>(b.height)};
}

template<typename _Tp1, typename _Tp2> inline
auto operator /= (cv::Size_<_Tp1>& a, cv::Size_<_Tp2> const& b) -> cv::Size_<_Tp1>&
{
    a.width /= b.width;
    a.height /= b.height;
    return a;
}

template<typename _Tp1, typename _Tp2> inline
auto operator * (cv::Size_<_Tp1> const& a, cv::Size_<_Tp2> const& b) -> cv::Size_<typename std::common_type<_Tp1,_Tp2>::type>
{
	typedef typename std::common_type<_Tp1,_Tp2>::type _Tp;
    return {static_cast<_Tp>(a.width)*static_cast<_Tp>(b.width), static_cast<_Tp>(a.height)*static_cast<_Tp>(b.height)};
}


template<typename _Tp1, typename _Tp2> inline
auto operator *= (cv::Size_<_Tp1>& a, cv::Size_<_Tp2> const& b) -> cv::Size_<_Tp1>&
{
    a.width *= b.width;
    a.height *= b.height;
    return a;
}



}


#endif // OPERATORS_HPP
