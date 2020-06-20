#include <cvextra/ComponentIterable.hpp>
#include <cvextra/improc.hpp>

auto cvx::details::ComponentIterator::
operator ++() -> ComponentIterator&
{
    ++iLabel;
    return *this;
}

auto cvx::details::ComponentIterator::
operator ==(
        const ComponentIterator& other) const -> bool
{
    return iLabel == other.iLabel;
}

auto cvx::details::ComponentIterator::
operator !=(
        const ComponentIterator& other) const -> bool
{
    return !((*this)==other);
}

auto cvx::details::ComponentIterator::
operator *() const -> cvx::BinaryMat
{
    return (labels == iLabel);
}

auto cvx::details::ComponentIterable::
begin() const -> ComponentIterator
{
    return {labels, 1};
}

auto cvx::details::ComponentIterable::
end() const -> ComponentIterator
{
    return {labels, nLabels};
}

auto cvx::
connectedComponentMasks(
        cvx::BinaryMat const& mat,
        int connectivity,
        int ddepth
        ) -> cvx::details::ComponentIterable
{
    cv::Mat1i labels;
    int nLabels = cvx::connectedComponents(mat, labels, connectivity, ddepth);

    return {labels, nLabels};

}
