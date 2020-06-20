#ifndef REGIONTEXTONHISTOGRAM_HPP
#define REGIONTEXTONHISTOGRAM_HPP

#include <cvextra/core.hpp>
#include <CrowdCounting/RegionCounting/Features/FeatureExtractor.hpp>
#include <CrowdCounting/RegionCounting/Features/PreprocessedFrame.hpp>
#include <stdx/cloning.hpp>
#include <memory>
#include <string>
#include <vector>

namespace crowd {

class RegionTextonHistogram : public FeatureExtractor
{
public:

    virtual auto extract(PreprocessedFrame const& frame, cvx::Rectd relativeRect) const -> std::vector<double>;
    virtual auto getFeatureCount() const -> int;
    virtual auto getNames() const -> std::vector<std::string>;
    virtual auto getDescription() const -> std::string;

    CVX_CLONE_IN_DERIVED(RegionTextonHistogram)
    CVX_CONFIG_DERIVED(RegionTextonHistogram)
};

}

#endif // AREA_HPP
