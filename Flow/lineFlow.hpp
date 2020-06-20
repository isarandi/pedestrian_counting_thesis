#ifndef LINEFLOW_HPP
#define LINEFLOW_HPP

#include <cvextra/core.hpp>
#include <cvextra/math.hpp>
#include <cvextra/improc.hpp>
#include <cvextra/ImageLoadingIterable.hpp>
#include <vector>

namespace crowd {
namespace lineopticalflow {

struct OpticalFlowOptions {
    double smoothnessAlpha;
    double magnitudeBeta;
    double epsilonSquare;
    double SORomega;
    int nScales;
    int nIterationsK;
    int nIterationsL;
    int nIterationsSOR;

    void writeToFile(cvx::bpath const& path);
    static auto fromFile(cvx::bpath const& path) -> OpticalFlowOptions;

};

auto solveOpticalFlow(
        std::vector<cv::Mat> const& images,
        std::vector<double> const& channelWeights,
        int iTargetImage,
        cv::Mat2d const& initialFlow,
        OpticalFlowOptions const& options
        ) -> cv::Mat2d;

//--- IMPLEMENTATION DETAILS
namespace details{

auto getRotatedRectExtractorMatrix(
        cv::RotatedRect rotatedRect
        ) -> cv::Mat;

auto solveAtScale(
        std::vector<cv::Mat> const& multiChannelImages,
        std::vector<double> const& channelWeights,
        int iTargetImage,
        cv::Mat2d const& initialFlow,
        OpticalFlowOptions const& options
        ) -> cv::Mat2d;

auto calcGradientFields(
        std::vector<std::vector<cv::Mat1d>> const& images,
        int exceptIndex
        ) -> std::vector<std::vector<cv::Mat2d> >;

auto solveConvexProblem(
        std::vector<std::vector<cv::Mat1d>> const& images,
        std::vector<double> const& channelWeights,
        int iTargetImage,
        cv::Mat2d const& flow,
        cv::Mat2d const& Ixx,
        cv::Mat1d const& Ix1x2,
        cv::Mat2d const& Ixdiff,
        OpticalFlowOptions const& options
        ) -> cv::Mat2d;

void successiveOverRelaxation(
        cv::Mat2d const& flow,
        cv::Mat2d& deltaFlow,
        cv::Mat2d const& Ixx,
        cv::Mat1d const& Ix1x2,
        cv::Mat2d const& Ixdiff,
        cv::Mat1d const& psiData,
        cv::Mat1d const& psiSmooth,
        cv::Mat1d const& psiMagnitude,
        OpticalFlowOptions const& options);

void linearizeDataTerm(
        std::vector<std::vector<cv::Mat1d>> const& images,
        std::vector<double> const& channelWeights,
        int iTargetImage,
        std::vector<std::vector<cv::Mat2d>> const& gradientFields,
        cv::Mat2d const& flow,
        cv::Mat2d& Ixx,
        cv::Mat1d& Ix1x2,
        cv::Mat2d& Ixdiff);

void calcPsiSmooth(
        cv::Mat2d const& flowPlusDeltaFlow,
        cv::Mat1d& diffusivity,
        double epsilonSq);

void calcPsiMagnitude(
        cv::Mat2d const& flowPlusDeltaFlow,
        cv::Mat1d& psiShort,
        double epsilonSq);

void calcPsiData(
        std::vector<std::vector<cv::Mat1d>> const& images,
        std::vector<double> const& channelWeights,
        int iTargetImage,
        cv::Mat2d const& flowPlusDeltaFlow,
        cv::Mat1d& psiData,
        double epsilonSq);

auto calcEnergy(
        std::vector<std::vector<cv::Mat1d>> const& images,
        std::vector<double> const& channelWeights,
        int iTargetImage,
        cv::Mat2d const& flow,
        OpticalFlowOptions const& options
        ) -> double;

} // end of namespace crowd::lineopticalflow::details

} // end of namespace crowd::lineopticalflow

} // end of namespace crowd


#endif // LINEFLOW_HPP
