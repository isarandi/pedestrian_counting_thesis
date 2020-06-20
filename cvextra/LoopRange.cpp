#include "LoopRange.hpp"
#include "coords.hpp"

using namespace cv;
using namespace cvx;
using namespace cvx::details;

auto PointRangeIterator::
operator ++() -> PointRangeIterator&
{
    ++p.x;
    if (p.x >= xEnd)
    {
        p.x = xBegin;
        ++p.y;
    }

    return *this;
}

auto PointRangeIterator::
operator ==(PointRangeIterator const& other) const -> bool
{
    return p == other.p;
}

auto PointRangeIterator::
operator !=(PointRangeIterator const& other) const -> bool
{
    return !(*this == other);
}

auto PointRangeIterator::
operator *() const -> Point const&
{
    return p;
}

auto PointRange::
begin() const -> PointRangeIterator
{
    return PointRangeIterator{Point{xBegin,yBegin}, xBegin,xEnd};
}

auto PointRange::
end() const -> PointRangeIterator
{
    return PointRangeIterator{Point{xBegin,yEnd}, xBegin,xEnd};
}

auto cvx::
points(Size size) -> PointRange
{
    return cvx::points(cvx::fullRect(size));
}

auto cvx::
points(InputArray arr) -> PointRange
{
    return cvx::points(cvx::fullRect(arr));
}

auto cvx::
points(Rect rect) -> PointRange
{
    return PointRange{rect.x, rect.x+rect.width, rect.y, rect.y+rect.height};
}

///////////// COLMAJOR //////////////////

auto ColMajorPointRangeIterator::
operator ++() -> ColMajorPointRangeIterator&
{
    ++p.y;
    if (p.y >= yEnd)
    {
        p.y = yBegin;
        ++p.x;
    }

    return *this;
}

auto ColMajorPointRangeIterator::
operator ==(ColMajorPointRangeIterator const& other) const -> bool
{
    return p == other.p;
}

auto ColMajorPointRangeIterator::
operator !=(ColMajorPointRangeIterator const& other) const -> bool
{
    return !(*this == other);
}

auto ColMajorPointRangeIterator::
operator *() const -> Point const&
{
    return p;
}

auto ColMajorPointRange::
begin() const -> ColMajorPointRangeIterator
{
    return ColMajorPointRangeIterator{Point{xBegin,yBegin}, yBegin,yEnd};
}

auto ColMajorPointRange::
end() const -> ColMajorPointRangeIterator
{
    return ColMajorPointRangeIterator{Point{xEnd,yBegin}, yBegin,yEnd};
}

auto cvx::
colMajorPoints(Size size) -> ColMajorPointRange
{
    return cvx::colMajorPoints(cvx::fullRect(size));
}

auto cvx::
colMajorPoints(InputArray arr) -> ColMajorPointRange
{
    return cvx::colMajorPoints(cvx::fullRect(arr));
}

auto cvx::
colMajorPoints(Rect rect) -> ColMajorPointRange
{
    return ColMajorPointRange{rect.x, rect.x+rect.width, rect.y, rect.y+rect.height};
}

auto cvx::
cvrange(Range const& range) -> decltype(boost::irange(0,1))
{
    return boost::irange(range.start, range.end);
}


