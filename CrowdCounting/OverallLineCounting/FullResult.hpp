#ifndef CROWDCOUNTING_OVERALLLINECOUNTING_FULLRESULT_HPP_
#define CROWDCOUNTING_OVERALLLINECOUNTING_FULLRESULT_HPP_

#include <CrowdCounting/OverallLineCounting/OverallLineCounter.hpp>
#include <CrowdCounting/RegionCounting/RegionCounter.hpp>
#include <MachineLearning/Regression.hpp>
#include <Python/Pyplot.hpp>
#include <opencv2/core/core.hpp>
#include <cvextra/filesystem.hpp>
#include <stdx/cloning.hpp>
#include <string>
#include <vector>

namespace crowd {
namespace linecounting {

struct ErrorCharacteristics
{
	double mean;
	double variance;

	auto getExpectedAbsError(double time) -> double;
	auto getExpectedAbsErrorCurve(int size) -> cv::Mat1d;
};

struct AssociationResult
{
    cv::Mat2d foundFlows;
    cv::Mat2d loneLookers;
};

struct Confusion
{
    double truePositives;
    double totalRetrieved;
    double totalRelevant;

    auto precision() const -> double {return truePositives/totalRetrieved;}
    auto recall() const -> double {return truePositives/totalRelevant;}
    auto f1() const -> double {
        double p = precision();
        double r = recall();
        return 2*p*r/(p+r);
    }

    auto operator+=(Confusion const& rhs) -> Confusion&
    {
        truePositives+=rhs.truePositives;
        totalRetrieved+=rhs.totalRetrieved;
        totalRelevant+=rhs.totalRelevant;
        return *this;
    }

    auto operator+(Confusion const& rhs) const -> Confusion
    {
        Confusion result = *this;
        result+=rhs;
        return result;
    }

};

class FullResult
{
public:
    FullResult(){}

    FullResult(
    		cv::Mat1d const& desiredRegionCounts,
			cv::Mat2d const& desiredLineFlow,
			PredictionWithConfidence const& predictedRegionCounts,
			PredictionWithConfidence2 const& predictedLineFlow,
			cv::Mat1d const& improvedRegionCounts,
			cv::Mat2d const& improvedLineFlow)
    	: desiredRegionCounts(desiredRegionCounts)
    	, desiredLineFlow(desiredLineFlow)
    	, predictedRegionCounts(predictedRegionCounts)
    	, predictedLineFlow(predictedLineFlow)
    	, improvedRegionCounts(improvedRegionCounts)
    	, improvedLineFlow(improvedLineFlow){}

    explicit
    FullResult(
            PredictionWithConfidence2 const& lineResult,
            cv::Mat2d const& desiredLineFlow)
        : desiredLineFlow(desiredLineFlow)
        , predictedLineFlow(lineResult)
        , desiredRegionCounts(cv::Mat1d::zeros(desiredLineFlow.rows, desiredLineFlow.cols-1))
        , predictedRegionCounts({
           cv::Mat1d::zeros(desiredLineFlow.rows, desiredLineFlow.cols-1),
           cv::Mat1d::ones(desiredLineFlow.rows, desiredLineFlow.cols-1)}){}

    auto associate(cv::Mat2d const& predicted, int maxDistance = 0) const -> AssociationResult;

    auto precisionRecallCurve(
            cv::Mat2d const& predicted,
            int maxDistance
            ) const -> cv::Mat1d;

    auto confusion(cv::Mat2d const& predicted, int distance) const -> Confusion;
    auto errorCharacteristics(cv::Mat2d const& predicted, int windowSize) const -> ErrorCharacteristics;

    auto lineFlowUncertaintyQuality() const -> double;
    auto regionUncertaintyQuality() const -> double;

    auto meanFinalAbsError(cv::Mat2d const& predicted) const -> double;
    auto meanFinalAbsRelError(cv::Mat2d const& predicted) const -> double;

    auto meanRegionAbsError(cv::Mat1d const& predicted) const -> double;
    auto meanRegionAbsRelError(cv::Mat1d const& predicted) const -> double;

    auto boxEvaluationCurve(cv::Mat2d const& predicted, bool averaged = true) const -> cv::Mat2d;

    auto plot(bool cumulative = true, bool show = true) const -> pyx::Pyplot;
    auto linePlot(bool cumulative = true, bool show = true) const -> pyx::Pyplot;
    auto regionPlot(bool uncertainty, bool show) const -> pyx::Pyplot;
    auto regionFromLinePlot(bool show) const -> pyx::Pyplot;

    static
    auto load(cvx::bpath const& path) -> FullResult;
    void save(cvx::bpath const& path) const;

    void horizontalAdd(FullResult const& r);

    CVX_CONFIG_SINGLE(FullResult);

    cv::Mat1d  desiredRegionCounts;
    cv::Mat2d  desiredLineFlow;

    PredictionWithConfidence  predictedRegionCounts;
    PredictionWithConfidence2  predictedLineFlow;

    cv::Mat1d  improvedRegionCounts;
    cv::Mat2d  improvedLineFlow;

    static
    auto horizontalMerge(std::vector<FullResult> const& results) -> FullResult;

private:
    mutable cvx::bpath folder;

};

}}



#endif /* CROWDCOUNTING_OVERALLLINECOUNTING_FULLRESULT_HPP_ */
