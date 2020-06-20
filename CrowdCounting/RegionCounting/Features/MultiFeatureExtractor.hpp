#ifndef COMPOSITEEXTRACTOR_HPP
#define COMPOSITEEXTRACTOR_HPP

#include <cvextra/core.hpp>
#include <CrowdCounting/RegionCounting/Features/FeatureExtractor.hpp>
#include <CrowdCounting/RegionCounting/Features/PreprocessedFrame.hpp>
#include <stdx/stdx.hpp>
#include <cvextra/configfile.hpp>
#include <initializer_list>
#include <memory>
#include <string>
#include <vector>

namespace crowd {

class MultiFeatureExtractor
{
public:
    MultiFeatureExtractor(
            std::vector<stdx::any_reference_wrapper<FeatureExtractor const>> extractors);
    MultiFeatureExtractor(
            std::initializer_list<stdx::any_reference_wrapper<FeatureExtractor const>> extractors);

    auto extractFeatures(
    		PreprocessedFrame const& frame,
			cvx::Rectd relativeRect
			) const -> std::vector<double>;

    auto preprocess(CountingFrame const& cf) const -> PreprocessedFrame;

    auto getFeatureCount() const -> int;
    auto getNames() const -> std::vector<std::string>;
    auto getDescription() const -> std::string;

    CVX_CONFIG_SINGLE(MultiFeatureExtractor)

    virtual ~MultiFeatureExtractor(){}

private:

    std::vector<std::shared_ptr<FeatureExtractor const>> extractors;
};

}

#endif // COMPOSITEEXTRACTOR_HPP
