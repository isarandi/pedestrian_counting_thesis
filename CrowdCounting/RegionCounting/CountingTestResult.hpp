#ifndef COUNTINGTESTRESULT_HPP
#define COUNTINGTESTRESULT_HPP

#include <MachineLearning/Regression.hpp>
#include <opencv2/core/core.hpp>
#include <memory>
#include <string>

namespace cvx
{
class Plotter;
} /* namespace cvx */

namespace crowd {

class CountingTestCase;

class CountingTestResult
{
public:
    CountingTestResult(
            CountingTestCase const& testCase,
            PredictionWithConfidence const& predictions);

    auto getMeanSquaredError() const -> double {return mse;}
    auto getMeanAbsoluteError() const -> double {return mae;}
    auto getMeanRelativeError() const -> double {return mre;}

    auto getTestCase() const -> CountingTestCase const& {return *testCaseClone;}
    auto getPredictions() const -> PredictionWithConfidence{return predictions;}
    auto getDesired() const -> cv::Mat1d{return desired;}

    auto getResultDescription() const -> std::string;

    void saveToLog() const;
    void saveImages() const;
    void saveMatlabCode() const;
    void savePlot() const;

private:
    std::shared_ptr<CountingTestCase> testCaseClone;
    PredictionWithConfidence predictions;
    cv::Mat1d desired;

    double mse;
    double mae;
    double mre;
    void calculateErrorMeasures();
};

}


#endif // COUNTINGTESTRESULT_HPP
