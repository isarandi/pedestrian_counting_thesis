#ifndef VECTORS_H
#define VECTORS_H

#include <opencv2/opencv.hpp>
#include <stddef.h>
#include <algorithm>
#include <cassert>
#include <vector>


namespace cvx { namespace vectors
{

template<typename T> inline
auto subVector(std::vector<T> const& v, int fromIndex, int toIndex) -> std::vector<T>
{
    return std::vector<T>(v.begin()+fromIndex, v.begin()+toIndex);
}

template<typename T> inline
auto subVector(std::vector<T> const& v, cv::Range const& range) -> std::vector<T>
{
    if (range == cv::Range::all())
    {
        return std::vector<T>(v.begin(), v.end());
    }
    return std::vector<T>(v.begin()+range.start, v.begin()+range.end);
}


template<typename T> inline
auto subVector(std::vector<T> const& v, int fromIndex) -> std::vector<T>
{
    return subVector(v, fromIndex, v.size());
}

template<typename T>
auto subVectorByRatio(std::vector<T> const& v, double from, double to) -> std::vector<T>
{
    return subVector(v, std::ceil(from * v.size()), std::ceil(to * v.size()));
}

template<typename T, typename Cont> inline
void push_back_all(std::vector<T> &dst, Cont const& elements)
{
    dst.insert(dst.end(), elements.begin(), elements.end());
}

template<typename T>
auto flatten(std::vector<std::vector<T> > const& v2D) -> std::vector<T>
{
    std::vector<T> res;
    for (auto const& v : v2D)
    {
        push_back_all(res, v);
    }
    return res;
}

template<typename T>
auto concat(std::vector<T> const& v1, std::vector<T> const& v2) -> std::vector<T>
{
    std::vector<T> result;
    result.reserve(v1.size()+v2.size());

    push_back_all(result, v1);
    push_back_all(result, v2);

    return result;
}

template<typename T>
auto range(T from, T to, T step=1) -> std::vector<T>
{
    std::vector<T> result;

    for (double v = from; v < to; v += step)
    {
        result.push_back(v);
    }
    return result;
}

template<typename T>
auto range(T to) -> std::vector<T>
{
    return range<T>(0, to);
}

template<typename T, typename UnaryFun>
auto transform(std::vector<T> const& vec, UnaryFun transformer)
-> std::vector<decltype(transformer(vec[0]))>
{
    std::vector<decltype(transformer(vec[0]))> result;
    result.reserve(vec.size());

    for (auto const& elem : vec)
    {
        result.push_back(transformer(elem));
    }
    return result;
}

template<typename T, typename Pred> inline
void removeIf(std::vector<T>& vec, Pred predicate)
{
    vec.erase(std::remove_if(vec.begin(), vec.end(), predicate), vec.end());
}

template <typename ForwardIterable>
auto fromIterable(ForwardIterable iterable) -> std::vector<decltype(*std::begin(iterable))>
{
    std::vector<decltype(*std::begin(iterable))> result;

    for (auto const& elem : iterable)
    {
        result.push_back(elem);
    }

    return result;
}

template <typename T, typename... Arg>
auto zipConstruct(std::vector<Arg> const&... argVec) -> std::vector<T>
{
    std::vector<size_t> sizes = std::vector<size_t>{(argVec.size())...};
    assert(std::all_of(sizes.begin(), sizes.end(), [&](size_t s){return s==sizes[0];}));
    size_t size = sizes[0];

    std::vector<T> result;
    result.reserve(size);

    for (size_t i = 0; i < size; ++i)
    {
        result.emplace_back((argVec[i])...);
    }

    return result;
}

template <typename Record, typename Attr>
auto project(
        std::vector<Record> const& records,
        Attr Record::*attribute
        ) -> std::vector<Attr>
{
    std::vector<Attr> result;
    result.reserve(records.size());

    for (auto&& record : records)
    {
        result.push_back(record.*attribute);
    }

    return result;
}

template <typename Out, typename Record, typename Attr>
auto project(
        std::vector<Record> const& records,
        Attr Record::*attribute
        ) -> std::vector<Out>
{
    std::vector<Out> result;
    result.reserve(records.size());

    for (auto&& record : records)
    {
        result.emplace_back(record.*attribute);
    }

    return result;
}

template <typename Record, typename Attr>
auto project(
        std::vector<Record> const& records,
        Attr (Record::*memberFunction) () const
        ) -> std::vector<Attr>
{
    std::vector<Attr> result;
    result.reserve(records.size());

    for (auto&& record : records)
    {
        result.push_back((record.*memberFunction)());
    }

    return result;
}

template <typename Out, typename Record, typename Attr>
auto project(
        std::vector<Record> const& records,
        Attr (Record::*memberFunction)() const) -> std::vector<Out>
{
    std::vector<Out> result;
    result.reserve(records.size());

    for (auto&& record : records)
    {
        result.emplace_back((record.*memberFunction)());
    }

    return result;
}

}}

#endif // VECTORS_HPP
