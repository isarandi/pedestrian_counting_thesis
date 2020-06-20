#ifndef CVEXTRA_COORDS_HPP_
#define CVEXTRA_COORDS_HPP_

#include <opencv2/core/core.hpp>
#include "cvextra/core.hpp"

namespace cvx {

auto intersect(cv::Rect r1, cv::Rect r2) -> cv::Rect;
auto intersect(cv::Range r1, cv::Range r2) -> cv::Range;

auto rel2abs(cv::Point2d relPoint, cv::Size size) -> cv::Point;
auto abs2rel(cv::Point absPoint, cv::Size size) -> cv::Point2d;

auto rel2abs(cvx::Rectd relRect, cv::Size size) -> cv::Rect;
auto abs2rel(cv::Rect absRect, cv::Size size) -> cvx::Rectd;

template <typename _Tp> inline
auto atRel(cv::Mat_<_Tp> const& mat, cv::Point2d relPos) -> _Tp const&
{ return mat(cvx::rel2abs(relPos, mat.size())); }

template <typename _Tp> inline
auto atRel(cv::Mat_<_Tp>& mat, cv::Point2d relPos) -> _Tp&
{ return mat(cvx::rel2abs(relPos, mat.size())); }

template <typename _Tp> inline
auto fullRect(cv::Size_<_Tp> size) -> cv::Rect_<_Tp>
{ return {0, 0, size.width, size.height}; }

auto fullRect(cv::InputArray input) -> cv::Rect;

template <typename _Tp> inline
auto center(cv::Rect_<_Tp> rect) -> cv::Point_<_Tp>
{ return {rect.x + rect.width/2, rect.y + rect.height/2}; }

template <typename _Tp> inline
auto center(cv::Size_<_Tp> size) -> cv::Point_<_Tp>
{ return {size.width/2, size.height/2}; }

auto center(cv::InputArray input) -> cv::Point;

auto borderInterpolate(
        cv::Point p,
        cv::Size size,
        int borderType
        ) -> cv::Point;

auto borderInterpolate(
        cv::Point p,
        cv::Size size,
        int borderTypeX,
        int borderTypeY
        ) -> cv::Point;

auto relativeGridRect(cv::Point gridIndices, cv::Size gridResolution) -> cvx::Rectd;

template<typename T1, typename T2> inline
auto contains(cv::Rect_<T1> const& container, cv::Rect_<T2> const& contained) -> bool
{
    return container.x <= contained.x
            && container.y <= contained.y
            && container.x+container.width >= contained.x+contained.width
            && container.y+container.height >= contained.y+contained.height;
}

template<typename T1, typename T2> inline
auto contains(cv::Rect_<T1> const& container, cv::Point_<T2> const& contained) -> bool
{
    return container.x <= contained.x
            && container.y <= contained.y
            && container.x+container.width > contained.x
            && container.y+container.height > contained.y;
}

template<typename T1, typename T2> inline
auto contains(cv::Size_<T1> const& container, cv::Point_<T2> const& contained) -> bool
{ return cvx::contains(cvx::fullRect(container), contained); }

template<typename T1, typename T2> inline
auto contains(cv::Size_<T1> const& container, cv::Rect_<T2> const& contained) -> bool
{ return cvx::contains(cvx::fullRect(container), contained); }

template<typename T> inline
auto contains(cv::InputArray container, cv::Point_<T> const& contained) -> bool
{ return cvx::contains(container.size(), contained); }

template<typename T> inline
auto contains(cv::InputArray container, cv::Rect_<T> const& contained) -> bool
{ return cvx::contains(container.size(), contained); }

auto contains(cv::Range const& range, int number) -> bool;
auto contains(cv::Range const& r1, cv::Range const& r2) -> bool;

#ifndef CVX_END
#define CVX_END
int const END = -1;
#endif
}

#endif /* CVEXTRA_COORDS_HPP_ */
