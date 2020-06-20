#ifndef OPTIMIZER_HPP
#define OPTIMIZER_HPP
#include "MachineLearning/HyperparameterOptimization/Experiment.hpp"

#include <opencv2/opencv.hpp>

#include <vector>
#include <string>

namespace crowd {

class ParameterConfig
{
public:
    double stepFactor;
    double initialValue;

    ParameterConfig(double stepFactor, double initialValue)
        : stepFactor(stepFactor)
        , initialValue(initialValue)
    {}

};

class ExperimentResult
{
public:
    ExperimentResult(
            std::vector<double> const& paramValues,
            double evaluationResult)
        : paramValues(paramValues)
        , evaluationResult(evaluationResult)
    {}

    bool areParamsAlmostEqual(std::vector<double> const& otherParamValues);

    std::vector<double> paramValues;
    double evaluationResult;
};

class Optimizer
{
public:
    Optimizer(Experiment* experiment, std::vector<ParameterConfig> const& parameters)
        : experiment(experiment)
        , parameters(parameters){}

    auto optimize() -> double;
    auto getBestParams() -> std::vector<double> {return paramValues;}

private:
    void evaluateCurrent();

    void optimize(int iParameter);
    auto optimize(int iParameter, double stepFactor) -> bool;
    auto optimizeInOneDirection(int iParameter, double stepFactor) -> bool;

    Experiment* experiment;
    std::vector<ParameterConfig> parameters;
    std::vector<double> paramValues;

    std::vector<ExperimentResult> previousResults;

    double currentError;
};

}

#endif // OPTIMIZER_HPP
