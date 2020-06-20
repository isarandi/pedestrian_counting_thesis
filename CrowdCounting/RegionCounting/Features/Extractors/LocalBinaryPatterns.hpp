#ifndef LOCALBINARYPATTERNS_HPP
#define LOCALBINARYPATTERNS_HPP

#include <cvextra/core.hpp>
#include <CrowdCounting/RegionCounting/Features/FeatureExtractor.hpp>
#include <CrowdCounting/RegionCounting/Features/PreprocessedFrame.hpp>
#include <opencv2/core/core.hpp>
#include <stdx/cloning.hpp>
#include <memory>
#include <string>
#include <vector>

namespace crowd {

class LocalBinaryPatterns : public FeatureExtractor
{
public:
    LocalBinaryPatterns(
            double radius,
            int nSamples,
            int nHistogramBins,
            cv::Size subGrid,
            bool masked = false)
        : radius(radius)
        , nSamples(nSamples)
        , nHistogramBins(nHistogramBins)
        , masked(masked)
        , subGrid(subGrid) {}

    virtual auto extract(PreprocessedFrame const& frame, cvx::Rectd relativeRect) const -> std::vector<double>;
    virtual auto getFeatureCount() const -> int;
    virtual auto getNames() const -> std::vector<std::string>;
    virtual auto getDescription() const -> std::string;

    CVX_CLONE_IN_DERIVED(LocalBinaryPatterns)
    CVX_CONFIG_DERIVED(LocalBinaryPatterns)

private:
    bool masked;
    int nHistogramBins;
    double radius;
    int nSamples;

    cv::Size subGrid;

};

}

#endif // LOCALBINARYPATTERNS_HPP
