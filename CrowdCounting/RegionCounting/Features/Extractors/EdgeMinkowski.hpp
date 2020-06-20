#ifndef EDGEMINKOWSKI_HPP
#define EDGEMINKOWSKI_HPP

#include <cvextra/core.hpp>
#include <CrowdCounting/RegionCounting/Features/FeatureExtractor.hpp>
#include <CrowdCounting/RegionCounting/Features/PreprocessedFrame.hpp>
#include <stdx/cloning.hpp>
#include <memory>
#include <string>
#include <vector>

namespace crowd {

class EdgeMinkowski : public FeatureExtractor
{
public:
    EdgeMinkowski(int maxRadius, bool masked = false)
        : maxRadius(maxRadius)
        , masked(masked){}

    virtual auto extract(PreprocessedFrame const& frame, cvx::Rectd relativeRect) const -> std::vector<double>;
    virtual auto getFeatureCount() const -> int;
    virtual auto getNames() const -> std::vector<std::string>;
    virtual auto getDescription() const -> std::string;

    CVX_CLONE_IN_DERIVED(EdgeMinkowski)
    CVX_CONFIG_DERIVED(EdgeMinkowski)

private:
    bool masked;
    int maxRadius;

};

}

#endif // EDGEMINKOWSKI_HPP
