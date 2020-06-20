#ifndef EDGEORIHISTOGRAM_HPP
#define EDGEORIHISTOGRAM_HPP

#include <cvextra/core.hpp>
#include <CrowdCounting/RegionCounting/Features/FeatureExtractor.hpp>
#include <CrowdCounting/RegionCounting/Features/PreprocessedFrame.hpp>
#include <opencv2/core/core.hpp>
#include <stdx/cloning.hpp>
#include <memory>
#include <string>
#include <vector>

namespace crowd {

class EdgeOriHistogram : public FeatureExtractor
{
public:
    EdgeOriHistogram(int nBins, bool masked = false);

    virtual auto extract(PreprocessedFrame const& frame, cvx::Rectd relativeRect) const -> std::vector<double>;
    virtual auto getFeatureCount() const -> int;
    virtual auto getNames() const -> std::vector<std::string>;
    virtual auto getDescription() const -> std::string;

    CVX_CLONE_IN_DERIVED(EdgeOriHistogram)
    CVX_CONFIG_DERIVED(EdgeOriHistogram)

private:
    int nBins;
    bool masked;

    int kernelSideLength;
    double horizontalSigma;
    double verticalSigma;

    std::vector<cv::Mat> filters;
};

}

#endif // EDGEORIHISTOGRAM_HPP
