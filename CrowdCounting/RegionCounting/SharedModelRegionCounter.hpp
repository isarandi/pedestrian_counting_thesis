#ifndef SHAREDMODELCROWDCOUNTER_HPP
#define SHAREDMODELCROWDCOUNTER_HPP

#include <CrowdCounting/RegionCounting/Features/MultiFeatureExtractor.hpp>
#include <CrowdCounting/RegionCounting/RegionCounter.hpp>
#include <MachineLearning/Regression.hpp>
#include <opencv2/core/core.hpp>
#include <stdx/cloning.hpp>
#include <memory>
#include <string>

namespace crowd {

/**
 * @brief Learns a single regression model to map a cell feature vector
 * to the people count in the cell.
 */
class SharedModelRegionCounter : public RegionCounter
{
public:
    SharedModelRegionCounter(
            cv::Size gridSize,
            MultiFeatureExtractor const& extractor,
            Regression const& regression,
            bool inverseScaleFeature = false)
        : gridSize(gridSize)
        , extractor(extractor)
        , regression(regression.clone())
        , inverseScaleFeature(inverseScaleFeature) {}

    // CrowdCounter interface
    virtual void train(FrameCollection const& trainingData);
    virtual auto predict(FrameCollection const& testData) const -> cv::Mat1d;
    virtual auto predictWithConfidence(
    		FrameCollection const& samples
			) const -> PredictionWithConfidence;

    virtual auto canGiveConfidence() const -> bool;

    virtual auto getGridSize() const -> cv::Size {return gridSize;}
    virtual void setGridSize(cv::Size s) {gridSize = s;}
    virtual auto getDescription() const -> std::string;

    CVX_CLONE_IN_DERIVED(SharedModelRegionCounter)
    CVX_CONFIG_DERIVED(SharedModelRegionCounter)

private:
    bool inverseScaleFeature;

    cv::Size gridSize;

    MultiFeatureExtractor extractor;
    stdx::cloned_unique_ptr<Regression> regression;

    auto getCaseID(FrameCollection const& sequence) const -> std::string;

    auto buildInputMatrix(FrameCollection const&  sequence) const -> cv::Mat;
    auto buildOutputMatrix(FrameCollection const& sequence) const -> cv::Mat;
};

}

#endif // SHAREDMODELCROWDCOUNTER_HPP
