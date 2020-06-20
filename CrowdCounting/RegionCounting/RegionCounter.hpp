#ifndef ABSTRACTCROWDCOUNTER_HPP
#define ABSTRACTCROWDCOUNTER_HPP

#include <opencv2/core/core.hpp>
#include <MachineLearning/Regression.hpp>
#include <stdx/cloning.hpp>
#include <cvextra/configfile.hpp>
#include <memory>
#include <string>

namespace crowd
{
class FrameCollection;
} /* namespace crowd */

namespace crowd {

/**
 * @brief Can be trained with annotated frames and then
 * used as a predictor for new frames.
 */
class RegionCounter
{
public:

    virtual void train(FrameCollection const& trainingData) = 0;

    /**
     * @return An NxM matrix where N is the number of test frames and M is
     * the number of grid cells for which this model predicts the counts.
     * The grid cells are linearized in a row-major order.
     */
    virtual auto predict(FrameCollection const& testData) const -> cv::Mat1d = 0;

    virtual auto predictWithConfidence(
    		FrameCollection const& samples
			) const -> PredictionWithConfidence = 0;
    virtual auto canGiveConfidence() const -> bool = 0;

    /**
     * @return Size of the grid for which this model predicts people counts.
     */
    virtual auto getGridSize() const -> cv::Size = 0;
    virtual void setGridSize(cv::Size s) = 0;
    virtual auto getDescription() const -> std::string = 0;

    virtual ~RegionCounter(){}

    CVX_CLONE_IN_BASE(RegionCounter)
    CVX_CONFIG_BASE(RegionCounter)
};

}
#endif // ABSTRACTCROWDCOUNTER_HPP
