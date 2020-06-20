#ifndef FEATUREEXTRACTOR_HPP
#define FEATUREEXTRACTOR_HPP

#include <CrowdCounting/RegionCounting/Features/PreprocessedFrame.hpp>
#include <cvextra/core.hpp>
#include <cvextra/configfile.hpp>
#include <stdx/cloning.hpp>
#include <string>
#include <vector>

namespace crowd {

class FeatureExtractor
{
public:

    virtual auto extract(
            PreprocessedFrame const& frame,
            cvx::Rectd relativeRect) const -> std::vector<double> = 0;

    virtual auto getFeatureCount() const -> int = 0;

    virtual auto getNames() const -> std::vector<std::string> = 0;
    virtual auto getDescription() const -> std::string = 0;

    virtual ~FeatureExtractor(){}
    CVX_CLONE_IN_BASE(FeatureExtractor)
    CVX_CONFIG_BASE(FeatureExtractor)
};

}

#endif // FEATUREEXTRACTOR_HPP
