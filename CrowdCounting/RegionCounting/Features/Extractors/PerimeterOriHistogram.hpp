#ifndef PERIMETERORIHISTOGRAM_HPP
#define PERIMETERORIHISTOGRAM_HPP

#include <cvextra/core.hpp>
#include <CrowdCounting/RegionCounting/Features/FeatureExtractor.hpp>
#include <CrowdCounting/RegionCounting/Features/PreprocessedFrame.hpp>
#include <opencv2/core/core.hpp>
#include <stdx/cloning.hpp>
#include <memory>
#include <string>
#include <vector>

namespace crowd {

class PerimeterOriHistogram : public FeatureExtractor
{
public:
    PerimeterOriHistogram(int nBins);

    virtual auto extract(PreprocessedFrame const& frame, cvx::Rectd relativeRect) const -> std::vector<double>;
    virtual auto getFeatureCount() const -> int;
    virtual auto getNames() const -> std::vector<std::string>;
    virtual auto getDescription() const -> std::string;

    CVX_CLONE_IN_DERIVED(PerimeterOriHistogram)
    CVX_CONFIG_DERIVED(PerimeterOriHistogram)

private:
    int nBins;

    int kernelSideLength;
    double horizontalSigma;
    double verticalSigma;

    std::vector<cv::Mat> filters;
};

}

#endif // PERIMETERORIHISTOGRAM_HPP
