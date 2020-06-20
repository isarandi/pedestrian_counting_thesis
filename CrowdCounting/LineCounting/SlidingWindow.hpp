#ifndef CROWDCOUNTING_LineCounting_SLIDINGWINDOW_HPP_
#define CROWDCOUNTING_LineCounting_SLIDINGWINDOW_HPP_

#include <cvextra/math.hpp>
#include <opencv2/core/core.hpp>
#include <functional>
#include <vector>


namespace crowd {
namespace linecounting
{


template <typename T>
class SlidingWindow {
public:
    SlidingWindow(){}

    SlidingWindow(T start, T size, T step)
        : start(start)
        , step(step)
        , size(size){}

    SlidingWindow(T size, T step)
            : start(static_cast<T>(0))
            , step(step)
            , size(size){}

    template <typename U>
    operator SlidingWindow<U> () const
    {
        return SlidingWindow<U>(
                static_cast<U>(start),
                static_cast<U>(size),
                static_cast<U>(step));
    }

    auto sectionCountOver(T overThisSizedInterval) const -> int
    {
        return static_cast<int>((overThisSizedInterval-size)/step)+1;
    }

    auto ith(int index) const -> cvx::Range_<T>
    {
        T rangeStart = start+index*step;
        return {rangeStart, rangeStart+size};
    }

//    typedef decltype(
//            boost::make_transform_iterator(
//                cvx::irange(0).begin(),
//                std::function<cvx::Range_<T>(int)>()))
//    iterator_t;
//
//    auto begin() -> iterator_t
//    {
//        boost::make_transform_iterator(
//                        cvx::irange(0).begin(),
//                        std::function<cvx::Range_<T>(int)>())
//    }
//
//    auto end() -> iterator_t
//    {
//
//    }
    T start;
    T size;
    T step;


};

template <typename T, typename U>
auto operator * (SlidingWindow<T> const& sw, U num) -> SlidingWindow<decltype(num*sw.start)>
{
    return {num*sw.start, num*sw.size, num*sw.step};
}

template <typename T, typename U>
auto operator * (U num, SlidingWindow<T> const& sw) -> SlidingWindow<decltype(num*sw.start)>
{
    return sw*num;
}

} /* namespace linecounting */
} /* namespace crowd */

#endif /* CROWDCOUNTING_LineCounting_SLIDINGWINDOW_HPP_ */
