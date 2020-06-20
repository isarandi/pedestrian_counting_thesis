#ifndef RANGE_HPP
#define RANGE_HPP
#include <opencv2/core/core.hpp>
#include <boost/range/irange.hpp>

namespace cvx {

namespace details {

struct PointRangeIterator
{
    auto operator ++() -> PointRangeIterator&;
    auto operator ==(PointRangeIterator const& other) const -> bool;
    auto operator !=(PointRangeIterator const& other) const -> bool;
    auto operator  *() const -> cv::Point const&;

    cv::Point p;
    int xBegin;
    int xEnd;
};

struct PointRange
{
    auto begin() const -> PointRangeIterator;
    auto end() const -> PointRangeIterator;

    int xBegin;
    int xEnd;
    int yBegin;
    int yEnd;
};

struct ColMajorPointRangeIterator
{
    auto operator ++() -> ColMajorPointRangeIterator&;
    auto operator ==(ColMajorPointRangeIterator const& other) const -> bool;
    auto operator !=(ColMajorPointRangeIterator const& other) const -> bool;
    auto operator  *() const -> cv::Point const&;

    cv::Point p;
    int yBegin;
    int yEnd;
};

struct ColMajorPointRange
{
    auto begin() const -> ColMajorPointRangeIterator;
    auto end() const -> ColMajorPointRangeIterator;

    int xBegin;
    int xEnd;
    int yBegin;
    int yEnd;
};


} // namespace details

auto points(cv::Size size) -> details::PointRange;
auto points(cv::InputArray arr) -> details::PointRange;
auto points(cv::Rect rect) -> details::PointRange;

auto colMajorPoints(cv::Size size) -> details::ColMajorPointRange;
auto colMajorPoints(cv::InputArray arr) -> details::ColMajorPointRange;
auto colMajorPoints(cv::Rect rect) -> details::ColMajorPointRange;

template<typename T> inline
auto irange(T from, T to, T step) -> decltype(boost::irange(from,to,step))
{
    if (to < from)
    {
        return boost::irange(from, from, 1);
    }

    int diff = to - from;
    if (diff % step != 0)
    {
        to = from+((diff/step)+1)*step;
    }
    return boost::irange(from, to, step);
}

template<typename T> inline
auto irange(T from, T to) -> decltype(boost::irange(from,to))
{
    if (to < from)
    {
        return boost::irange(from, from);
    }
    return boost::irange(from, to);
}

template<typename T> inline
auto irange(T to) -> decltype(boost::irange(static_cast<T>(0),to))
{
    if (to < 0)
    {
        return boost::irange(static_cast<T>(0), static_cast<T>(0));
    }
    return boost::irange(static_cast<T>(0), to);
}

auto cvrange(cv::Range const& range) -> decltype(boost::irange(0,1));

} // namespace cvx

#endif // RANGE_HPP
