#ifndef MULTIVARIATEREGRESSION_HPP
#define MULTIVARIATEREGRESSION_HPP

#include <MachineLearning/LearningSet.hpp>
#include <opencv2/core/core.hpp>
#include <cvextra/configfile.hpp>
#include <stdx/cloning.hpp>
#include <memory>
#include <string>
#include <vector>


namespace crowd {

class Regression
{
public:
    virtual void train(LearningSet const& ls) = 0;
    virtual auto predict(cv::Mat1d const& X) const -> cv::Mat1d = 0;

    virtual auto getJacobian(cv::Mat1d const& singleX) const -> cv::Mat1d = 0;

    virtual auto getDescription() const -> std::string = 0;


    virtual ~Regression(){}

    CVX_CLONE_IN_BASE(Regression)
    CVX_CONFIG_BASE(Regression)
};

struct PredictionWithConfidence
{
    cv::Mat1d mean;
    cv::Mat1d variance;
};

struct PredictionWithConfidence2
{
    cv::Mat2d mean;
    cv::Mat2d variance;
};

class RegressionWithConfidence : public Regression
{
public:
    virtual auto predictWithConfidence(cv::Mat1d const& X) const -> PredictionWithConfidence = 0;
    virtual ~RegressionWithConfidence(){}

    CVX_CLONE_IN_BASE(RegressionWithConfidence)
    CVX_CONFIG_BASE(RegressionWithConfidence)
};


}

#endif // MULTIVARIATEREGRESSION_HPP
