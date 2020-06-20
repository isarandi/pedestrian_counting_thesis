#ifndef CROWDCOUNTER_HPP
#define CROWDCOUNTER_HPP

#include <CrowdCounting/RegionCounting/Features/MultiFeatureExtractor.hpp>
#include <CrowdCounting/RegionCounting/RegionCounter.hpp>
#include <MachineLearning/Regression.hpp>
#include <opencv2/core/core.hpp>
#include <stdx/cloning.hpp>
#include <memory>
#include <string>

namespace crowd {

/**
 * @brief Learns the mapping from the concatenated feature vector extracted from
 * all grid cells to the concatenated number of people in each grid cell.
 */
class AllToAllRegionCounter : public RegionCounter
{
public:
    AllToAllRegionCounter(
            cv::Size gridSize,
            MultiFeatureExtractor const& extractor,
            Regression const& regression)
        : gridSize(gridSize)
        , extractor(extractor)
        , regression(regression.clone()){}

    virtual void train(FrameCollection const& trainingData);
    virtual auto predict(FrameCollection const& testData) const -> cv::Mat1d;
    virtual auto predictWithConfidence(
    		FrameCollection const& samples
			) const -> PredictionWithConfidence;

    virtual auto canGiveConfidence() const -> bool;

    virtual void setGridSize(cv::Size s) {gridSize = s;}
    virtual auto getGridSize() const -> cv::Size {return gridSize;}
    virtual auto getDescription() const -> std::string;

    auto buildInputMatrix(FrameCollection const& sequence) const -> cv::Mat1d;
    auto buildOutputMatrix(FrameCollection const& sequence) const -> cv::Mat1d;

    CVX_CLONE_IN_DERIVED(AllToAllRegionCounter)
    CVX_CONFIG_DERIVED(AllToAllRegionCounter)

private:
    cv::Size gridSize;
    MultiFeatureExtractor extractor;
    stdx::cloned_unique_ptr<Regression> regression;

    auto getCaseID(FrameCollection const& sequence) const -> std::string;
};

}

#endif // CROWDCOUNTER_HPP
