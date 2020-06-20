#ifndef CVEXTRA_COMPONENTITERABLE_HPP_
#define CVEXTRA_COMPONENTITERABLE_HPP_

#include <cvextra/core.hpp>

namespace cvx {

namespace details {

struct ComponentIterator
{
    auto operator ++() -> ComponentIterator&;
    auto operator ==(ComponentIterator const& other) const -> bool;
    auto operator !=(ComponentIterator const& other) const -> bool;
    auto operator  *() const -> cvx::BinaryMat;

    cv::Mat labels;
    int iLabel;
};

struct ComponentIterable
{
    auto begin() const -> ComponentIterator;
    auto end() const -> ComponentIterator;

    cv::Mat labels;
    int nLabels;
};

} // namespace details

auto connectedComponentMasks(
        cvx::BinaryMat const& mat,
        int connectivity,
        int ddepth = CV_32S
        ) -> details::ComponentIterable;

} // namespace cvx

#endif /* CVEXTRA_COMPONENTITERABLE_HPP_ */
