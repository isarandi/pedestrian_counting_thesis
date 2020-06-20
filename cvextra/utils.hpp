#ifndef UTILS_HPP
#define UTILS_HPP

#include <stddef.h>
#include <algorithm>
#include <cassert>
#include <functional>
#include <queue>
#include <string>
#include <utility>
#include <vector>

namespace cvx
{




auto timestamp() -> std::string;

//namespace details
//{
//    class ParallelBodyWithLambda : public cv::ParallelLoopBody
//    {
//    public:
//        ParallelBodyWithLambda(std::function<void(cv::Range)> function)
//            : function(function){}

//        void operator ()(cv::Range const& range) const
//        {
//            function(range);
//        }
//    private:
//        std::function<void(cv::Range)> function;
//    };
//}

//void parallel_for_(cv::Range range, std::function<void(cv::Range)> function);

template <typename T, typename... Arg>
std::vector<T> zipConstruct(std::vector<Arg> const&... argVec)
{
    std::vector<size_t> sizes = {argVec.size()...};
    assert(std::all_of(sizes.begin(), sizes.end(), [&](size_t const s){return s==sizes[0];}));
    size_t size = sizes[0];

    std::vector<T> result;
    result.reserve(size);

    for (size_t i = 0; i < size; ++i)
    {
        result.emplace_back((argVec[i])...);
    }

    return result;
}

template <typename T>
class top_queue
{
public:
    top_queue(int maxSize)
        : maxSize(maxSize)
        , q(&compare){}

    void push(T const& t, double value)
    {
        if (q.size() == maxSize && value < q.top().second)
        {
            return;
        }

        q.push(std::make_pair(t, value));

        if (q.size() > maxSize)
        {
            q.pop();
        }
    }

    auto popAll() -> std::vector<T>
    {
        std::vector<T> vec;
        while (!q.empty())
        {
            vec.push_back(q.top().first);
            q.pop();
        }

        std::reverse(vec.begin(), vec.end());
        return vec;
    }

    auto size() -> int {return q.size();}

private:
    int maxSize;

    static
    bool compare(std::pair<T, double> const& lhs, std::pair<T, double> const& rhs)
    {
        return lhs.second > rhs.second;
    };

    std::priority_queue<std::pair<T, double>, std::vector<std::pair<T, double>>, decltype(&compare)> q;
};

}

#endif // UTILS_HPP
