#ifndef MACHINELEARNING_NORMALIZEDREGRESSIONWITHCONFIDENCE_HPP_
#define MACHINELEARNING_NORMALIZEDREGRESSIONWITHCONFIDENCE_HPP_

#include <MachineLearning/LearningSet.hpp>
#include <MachineLearning/Regression.hpp>
#include <MachineLearning/DataNormalizer.hpp>
#include <opencv2/core/core.hpp>
#include <stdx/cloning.hpp>
#include <memory>
#include <string>
#include <vector>

namespace crowd {

class NormalizedRegressionWithConfidence : public RegressionWithConfidence
{
public:
    NormalizedRegressionWithConfidence(
            RegressionWithConfidence const& innerRegression,
            DataNormalizer const& normalizer = PerFeatureNormalizer())
        : innerRegression(innerRegression)
        , xNormalizer(normalizer){}

    virtual void
    train(LearningSet const& ls);

    virtual auto
    predict(cv::Mat1d const& X) const -> cv::Mat1d;

    virtual auto
    predictWithConfidence(cv::Mat1d const& X) const -> PredictionWithConfidence;

    virtual auto
    getJacobian(cv::Mat1d const& X) const -> cv::Mat1d;

    virtual auto
    getDescription() const -> std::string;

    virtual auto
    describe() const -> boost::property_tree::ptree;

    static auto
    create(boost::property_tree::ptree const& pt)
        -> std::unique_ptr<NormalizedRegressionWithConfidence>;

    CVX_CLONE_IN_DERIVED(NormalizedRegressionWithConfidence)

private:
    stdx::cloned_unique_ptr<DataNormalizer> xNormalizer;
    cv::Mat1d ymeans; //row mat

    stdx::cloned_unique_ptr<RegressionWithConfidence> innerRegression;
};

}

#endif /* MACHINELEARNING_NORMALIZEDREGRESSIONWITHCONFIDENCE_HPP_ */
