#ifndef IMREADITERABLE_HPP
#define IMREADITERABLE_HPP

#include "cvextra/filesystem.hpp"
#include "cvextra/coords.hpp"
#include <opencv2/core/core.hpp>
#include <boost/iterator/transform_iterator.hpp>
#include <boost/regex.hpp>
#include <functional>
#include <vector>

namespace cvx {

class ImageLoadingIterable
{

public:
    typedef decltype(
            boost::make_transform_iterator(
                std::vector<bpath>().cbegin(),
                std::function<cv::Mat(bpath const&)>()))
    iterator_t;

    ImageLoadingIterable(std::vector<bpath> const& paths, int flags=1);
    ImageLoadingIterable(bpath const& folderPath, boost::regex const& pattern=boost::regex(".*"), int flags=1);

    auto operator[](int i) const -> cv::Mat;

    auto begin() const -> iterator_t;
    auto end() const -> iterator_t;

    auto size() const -> size_t;
    auto getPaths() const -> std::vector<bpath>{return paths;}

    auto load() const -> std::vector<cv::Mat>;

    auto range(int from, int to=cvx::END) const -> ImageLoadingIterable;
    auto range(cv::Range const& range) const -> ImageLoadingIterable;

private:
    std::vector<bpath> paths;
    int flags;

};

auto imagesIn(bpath const& folderPath, std::string const& pattern=".*", int flags=1) -> ImageLoadingIterable;
auto imagesIn(std::vector<bpath> const& folderPath, std::string const& pattern=".*", int flags=1) -> ImageLoadingIterable;

auto loadImages(bpath const& folderPath, std::string const& pattern=".*", int flags=1) -> std::vector<cv::Mat>;

template <typename Iterator>
class Iterable {

public:
    Iterable(Iterator itBegin, Iterator itEnd)
        : itBegin(itBegin)
        , itEnd(itEnd){}

    auto begin() -> Iterator {return itBegin;}
    auto end() -> Iterator {return itEnd;}

    auto begin() const -> Iterator const {return itBegin;}
    auto end() const -> Iterator const {return itEnd;}

private:
    Iterator itBegin;
    Iterator itEnd;
};

template <typename Iterator>
auto iterable(Iterator itBegin, Iterator itEnd)
-> Iterable<Iterator>
{
    return {itBegin, itEnd};
}

template <typename IterableType>
auto subiterable(IterableType const& iterable, int fromIndex, int toIndex=-1)
-> decltype(cvx::iterable(std::begin(iterable), std::end(iterable)))
{
    using namespace std;
    auto itBegin = std::next(begin(iterable), fromIndex);
    auto itEnd = (toIndex == -1) ? end(iterable) : std::next(begin(iterable), toIndex);

    return cvx::iterable(itBegin, itEnd);
}


}

#endif // IMREADITERABLE_HPP
