#ifndef STATISTICALLANDSCAPE_HPP
#define STATISTICALLANDSCAPE_HPP

#include "CrowdCounting/RegionCounting/Features/FeatureExtractor.hpp"

namespace crowd {

class StatisticalLandscape : public FeatureExtractor
{
public:
    StatisticalLandscape(int nThresholds, bool masked = false);
    StatisticalLandscape(std::vector<double> const& thresholds, bool masked = false);

    virtual auto extract(PreprocessedFrame const& frame, cvx::Rectd relativeRect) const -> std::vector<double>;
    virtual auto getFeatureCount() const -> int;
    virtual auto getNames() const -> std::vector<std::string>;
    virtual auto getDescription() const -> std::string;

    CVX_CLONE_IN_DERIVED(StatisticalLandscape)
    CVX_CONFIG_DERIVED(StatisticalLandscape)

private:
    bool masked;
    std::vector<double> thresholds;
};

}

#endif // STATISTICALLANDSCAPE_HPP
