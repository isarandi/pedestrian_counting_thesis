#ifndef RUN_LINECOUNTING_LINECOUNTINGEXPERIMENT_HPP_
#define RUN_LINECOUNTING_LINECOUNTINGEXPERIMENT_HPP_

#include <CrowdCounting/LineCounting/LineCountingSet.hpp>
#include <CrowdCounting/LineCounting/LineCountingExperiment.hpp>
#include <cvextra/configfile.hpp>
#include <CrowdCounting/LineCounting/LineCounter.hpp>
#include <opencv2/core/core.hpp>
#include <string>


namespace crowd { namespace linecounting {

class LineCountingScenario {
public:
    std::vector<LineCountingSet> trainings;
    std::vector<LineCountingSet> tests;
    CVX_CONFIG_SINGLE(LineCountingScenario)
};

class LineCountingResult
{
public:
    LineCountingSet countingSet;
    cv::Mat1d desiredPerFrame;
    cv::Mat1d desiredPerWindow;
    PredictionWithConfidence predictionPerFrame;
    PredictionWithConfidence predictionPerWindow;

    auto boxEvaluationCurve() const ->  cv::Mat1d;

    static auto boxEvaluationCurve(
            cv::Mat2d prediction,
            cv::Mat2d validation
            ) -> cv::Mat1d;

    auto getDriftSigma() const -> double;

    //CVX_CONFIG_SINGLE(LineCountingResult)
};

class LineCountingTestCase
{
public:
    LineCountingScenario scenario;
    LineCounter flowCounter;

    auto run() -> std::vector<LineCountingResult>;
};


}}

#endif /* RUN_LINECOUNTING_LINECOUNTINGEXPERIMENT_HPP_ */
