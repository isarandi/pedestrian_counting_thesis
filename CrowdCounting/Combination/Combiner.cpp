#include <CrowdCounting/Combination/Combiner.hpp>
#include <CrowdCounting/RegionCounting/CountingFrame.hpp>
#include <CrowdCounting/RegionCounting/CountingTestCase.hpp>
#include <CrowdCounting/RegionCounting/CountingTestResult.hpp>
#include <CrowdCounting/RegionCounting/FrameCollection.hpp>
#include <CrowdCounting/PersonLocations.hpp>
#include <cvextra/cvenums.hpp>
#include <cvextra/cvret.hpp>
#include <cvextra/io.hpp>
#include <cvextra/ImageLoadingIterable.hpp>
#include <cvextra/LoopRange.hpp>
#include <cvextra/math.hpp>
#include <cvextra/mats.hpp>
#include <opencv2/core/mat.hpp>
#include <opencv2/core/operations.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <Python/EasyCallable.hpp>
#include <Python/EasyObject.hpp>
#include <Python/Pyplot.hpp>
#include <Run/config.hpp>
#include <CrowdCounting/LineCounting/LineCountingExperiment.hpp>
#include <cmath>
#include <iostream>
#include <set>
#include <utility>

using namespace crowd::linecounting;
using namespace cv;
using namespace cvx;
using namespace std;

auto crowd::linecounting::
aggregateLinesToRegionsInbetween(Mat2d const& c) -> Mat1d
{
    Mat q = c.colRange(1, c.cols)-c.colRange(0, c.cols-1);
    q = q.reshape(1, q.rows*q.cols);
    q.col(1)*=-1;
    return cvret::reduce(q, 1, REDUCE_SUM).reshape(1, c.rows);
}

auto getLineBasedRegionCounts(RegionAndLineCounts const& n) -> Mat1d
{
    return cvxret::cumsum(aggregateLinesToRegionsInbetween(n.l), 0);
}

static void plot(
        Mat2d const& c,
        Mat1d const& a,
        RegionAndLineCounts const& origPred,
        RegionAndLineCounts const& desired)
{
    pyx::Pyplot plt;

    auto fig_axarr = plt.subplots({4,origPred.l.cols}, "sharey='row',figsize=(20,15),squeeze=False");
    auto fig = fig_axarr[0];
    auto axarr = fig_axarr[1];

    Mat lineBasedRegionEstimate = aggregateLinesToRegionsInbetween(c) + cv::repeat(a, c.rows, 1);

    Mat1d cumulLeft = cvret::extractChannel(c, 0);
    Mat1d cumulRight = cvret::extractChannel(c, 1);

    Mat1d cumulLeftO = cvxret::cumsum(cvret::extractChannel(origPred.l, 0),0);
    Mat1d cumulRightO = cvxret::cumsum(cvret::extractChannel(origPred.l, 1),0);

    Mat1d cumulLeftV = cvxret::cumsum(cvret::extractChannel(desired.l, 0),0);
    Mat1d cumulRightV = cvxret::cumsum(cvret::extractChannel(desired.l, 1),0);

    Mat1d left = cvxret::backwardDerivative(cvret::extractChannel(c, 0));
    Mat1d right = cvxret::backwardDerivative(cvret::extractChannel(c, 1));

    Mat1d leftO = cvret::extractChannel(origPred.l, 0);
    Mat1d rightO = cvret::extractChannel(origPred.l, 1);

    Mat1d leftV = cvret::extractChannel(desired.l, 0);
    Mat1d rightV = cvret::extractChannel(desired.l, 1);

    Mat1d regiono = getLineBasedRegionCounts(origPred);
    Mat1d regiong = getLineBasedRegionCounts(desired);

    for (int iLine : cvx::irange(origPred.l.cols))
    {
        axarr[1][iLine].call("plot", {cumulLeftO.col(iLine), "k"});
        axarr[1][iLine].call("plot", {cumulLeft.col(iLine), "r"});
        axarr[1][iLine].call("plot", {cumulLeftV.col(iLine), "g"});

        axarr[2][iLine].call("plot", {cumulRightO.col(iLine), "k"});
        axarr[2][iLine].call("plot", {cumulRight.col(iLine), "r"});
        axarr[2][iLine].call("plot", {cumulRightV.col(iLine), "g"});

        axarr[3][iLine].call("plot", {cumulLeftO.col(iLine)-cumulRightO.col(iLine), "k"});
        axarr[3][iLine].call("plot", {cumulLeft.col(iLine)-cumulRight.col(iLine), "r"});
        axarr[3][iLine].call("plot", {cumulLeftV.col(iLine)-cumulRightV.col(iLine), "g"});

        axarr[1][iLine].call("plot", {leftO.col(iLine), "k"});
        axarr[1][iLine].call("plot", {left.col(iLine), "r"});
        axarr[1][iLine].call("plot", {leftV.col(iLine), "g"});

        axarr[2][iLine].call("plot", {rightO.col(iLine), "k"});
        axarr[2][iLine].call("plot", {right.col(iLine), "r"});
        axarr[2][iLine].call("plot", {rightV.col(iLine), "g"});

        axarr[3][iLine].call("plot", {leftO.col(iLine)-rightO.col(iLine), "k"});
        axarr[3][iLine].call("plot", {left.col(iLine)-right.col(iLine), "r"});
        axarr[3][iLine].call("plot", {leftV.col(iLine)-rightV.col(iLine), "g"});

        if (iLine < origPred.r.cols)
        {
            axarr[0][iLine].call("plot",{lineBasedRegionEstimate.col(iLine).clone(), "r"});
            axarr[0][iLine].call("plot",{origPred.r.col(iLine).clone(), "k"});
            axarr[0][iLine].call("plot",{desired.r.col(iLine).clone(), "g"});
            axarr[0][iLine].call("plot",{regiono.col(iLine).clone(), "blue"});
            axarr[0][iLine].call("plot",{regiong.col(iLine).clone(), "cyan"});
        }
    }

    plt.showAndClose();
}

static auto calcEnergy(
        Mat2d c,
        Mat1d a,
        RegionAndLineCounts const& o,
        Mat2d alpha,
        Mat1d gamma,
        double rho
        ) -> double
{
    Mat cderiv = cvxret::backwardDerivative(c);


    double sumSqLdiff = cvx::FrobeniusSq(cvret::sqrt(alpha).mul(cderiv - o.l));

    Mat regionEstimates = aggregateLinesToRegionsInbetween(c) + cv::repeat(a, c.rows, 1);

    double sumSqGhostCount = cvx::FrobeniusSq(cvret::sqrt(gamma).mul(regionEstimates - o.r));
    double sumSqS = cvx::FrobeniusSq(a);

    cout << "sumSqGhostCount = " << sumSqGhostCount << endl;
    cout << "sumSqLdiff = " << sumSqLdiff << endl;
    cout << "sumSqS = " << sumSqS << endl;

    cout << "rho = " << rho << endl;

    return sumSqLdiff + sumSqGhostCount + rho*sumSqS;
}

class NanDog {
public:
    std::string firstNanName;
    cv::Mat firstNanMat;
    bool hasFoundNan = false;

    void check(Mat const& m, std::string const& name)
    {
        if (!hasFoundNan && std::any_of(m.begin<double>(), m.end<double>(), [](double d){return !std::isnormal(d);}))
        {
            firstNanName = name;
            firstNanMat = m.clone();
            hasFoundNan = true;
        }
    }

    void check(double d, std::string const& name)
    {
        if (!hasFoundNan && std::isnan(d))
        {
            firstNanName = name;
            hasFoundNan = true;
        }
    }
};

auto isAllNormal(Mat m) -> bool
{
    return std::all_of(
            m.begin<double>(),
            m.end<double>(),
            [](double d)
            {
                return std::isnormal(d);
            });
}

auto crowd::linecounting::
solveCombinationOnSingleScale(
        RegionAndLineCounts const& o,
        RegionAndLineCounts const& g,
        OptimizationState const& initialState,
        Mat2d const& _alpha,
		double aprioriOutput,
		double aprioriDeviationReward,
		Mat1d const& gamma,
        double rho,
        double SORomega,
        int nSORiterations
        )  -> OptimizationState
{
    Mat2d c = initialState.c.clone();
    Mat1d a = initialState.a.clone();

    int nLines = o.l.cols;
    int nRegions = o.r.cols;
    int nFrames = o.l.rows;

    Mat1d meanReferenceRegionEstimates =
            cvret::reduce(o.r, 0, REDUCE_AVG);

    Mat2d lderiv = cvxret::forwardDerivative(o.l);
    Mat2d alpha = _alpha;
    //alpha = Vec2d{100,100};
    Mat2d alphaDeriv = cvxret::backwardDerivative(alpha);
    alphaDeriv.row(0).setTo(0);

    NanDog nanDog;

    for (int iSOR : cvx::irange(nSORiterations))
    {
        if ((iSOR%1000)==0)
        {
            cout << calcEnergy(c,a, o,alpha, gamma, rho) << endl;
            //plot(c, a, o, g);
        }

        {
            nanDog.check(c, "c_up_"+std::to_string(iSOR));
            Mat1d meanNewRegionEstimates =
                    cvret::reduce(aggregateLinesToRegionsInbetween(c), 0, REDUCE_AVG);

            nanDog.check(meanNewRegionEstimates, "meanNewRegionEstimates");

            double gamma = 0.1;
            Mat1d d = gamma * (meanNewRegionEstimates - meanReferenceRegionEstimates);

            nanDog.check(d, "d");
            double A = gamma + rho/nFrames;
            nanDog.check(A, "A");
            a = -d/A;//(1-SORomega) * a + SORomega * (-d/A);

            nanDog.check(a, "a");
        }

        for (int t : cvx::irange(nFrames))
        {
            for (int i : cvx::irange(nLines))
            {
                double qIMinus1Diff =
                        (i > 0) ? gamma(t,i-1) * ( a(0,i-1) + c(t,i-1)[1]-c(t,i-1)[0] - o.r(t,i-1) ) : 0;

                double qIDiff =
                        (i < nRegions) ? gamma(t,i) * ( a(0,i) + c(t,i+1)[0] - c(t,i+1)[1] - o.r(t,i) ) : 0;
//
//                double qIMinus1DiffApri =
//                        (i > 0) ? a(0,i-1) + c(t,i-1)[1]-c(t,i-1)[0] : 0;
//                double qIDiffApri =
//                        (i < o.r.cols) ? a(0,i) + c(t,i+1)[0] - c(t,i+1)[1] : 0;

                double di = -qIDiff + qIMinus1Diff +
                        -(alpha(t,i)[0])*(((t<nFrames-1) ? c(t+1,i)[0] : 0) + ((t>0) ? c(t-1,i)[0] : 0))
                        +alpha(t,i)[0] * lderiv(t,i)[0]
                        +alphaDeriv(t,i)[0]*(o.l(t,i)[0] + ((t>0) ? c(t-1,i)[0] : 0));

                double dj = qIDiff - qIMinus1Diff +
                        -(alpha(t,i)[1])*(((t<nFrames-1) ? c(t+1,i)[1] : 0) + ((t>0) ? c(t-1,i)[1] : 0))
                        +alpha(t,i)[1] * lderiv(t,i)[1]
                        +alphaDeriv(t,i)[1]*(o.l(t,i)[1] + ((t>0) ? c(t-1,i)[1] : 0));


                double rg = (i>0 ? gamma(t,i-1) : 0) + (i<nRegions ? gamma(t,i) : 0);
                int nNeighborTimes = (t<nFrames-1) ? 2 : 1;

                Matx22d A{
                        nNeighborTimes*(alpha(t,i)[0]) + rg - alphaDeriv(t,i)[0],
                        -rg, -rg,
                        nNeighborTimes*(alpha(t,i)[1]) + rg - alphaDeriv(t,i)[1]};

                Vec2d v = cvx::vec(A.solve(-Matx21d{di,dj}));

                if (std::isnan(v[0]) || std::isnan(v[1]))
                {
                    cout << "gamma(t,i) " << gamma(t,i)<< endl;
                    cout << gamma.rows << " " << gamma.cols << endl;

                    cout << "vnan" << endl;

                    throw 1;
                }


                c(t,i) = (1.0 - SORomega) * c(t,i) + SORomega * v;

            }
        }

    }

    cout << nanDog.firstNanName << endl;

    return {c,a};
}

#define PRINT_NAMED(x) std::cout << #x " = " << x << std::endl

auto crowd::linecounting::
solveCombinationCoarseToFine(
        RegionAndLineCounts predictedCounts,
        RegionAndLineCounts const& desiredCounts,
        cv::Mat2d const& alpha,
		double aprioriOutput,
		double aprioriDeviationReward,
		cv::Mat1d const& gamma,
        double rho,
        double SORomega,
        int nSORiterations
        )  -> RegionAndLineCounts
{
    int nRegions = predictedCounts.r.cols;
    int nLines = predictedCounts.l.cols;
    int nFrames = predictedCounts.r.rows;

    OptimizationState os {predictedCounts.l.clone(), Mat1d::zeros(1, nRegions)};

    double scaledIdealHeight = nFrames;
    int nScales = (int)std::log2(nFrames/3.0);

    vector<RegionAndLineCounts> scaledOrigPred{predictedCounts};
    vector<RegionAndLineCounts> scaledDesired{desiredCounts};
    vector<Mat2d> alphas{alpha};
    vector<Mat1d> gammas{gamma};

    for (int iScale : cvx::irange(1,nScales))
    {
        scaledIdealHeight *= 0.5;
        int downScaledHeight = (int)scaledIdealHeight;
        double downScalingFactor = downScaledHeight / (double)scaledOrigPred.back().r.rows;

        scaledOrigPred.push_back(
                {
                    cvret::resize(scaledOrigPred.back().r, Size{nRegions,downScaledHeight}, 1, 0.5, INTER_AREA),
                    (1./downScalingFactor)*cvret::resize(scaledOrigPred.back().l, Size{nLines,downScaledHeight}, 1, 0.5, INTER_AREA)});

        scaledDesired.push_back(
                {
                    cvret::resize(scaledDesired.back().r, Size{nRegions,downScaledHeight}, 1, 0.5, INTER_AREA),
                    (1./downScalingFactor)*cvret::resize(scaledDesired.back().l, Size{nLines,downScaledHeight}, 1, 0.5, INTER_AREA)});

        cv::resize(os.c, os.c, Size{nLines,downScaledHeight}, 1, 0.5, INTER_AREA);
        alphas.push_back(cvret::resize(alphas.back(), Size{nLines,downScaledHeight}, 1, 0.5, INTER_AREA)*downScalingFactor);
        gammas.push_back(cvret::resize(gammas.back(), Size{nRegions,downScaledHeight}, 1, 0.5, INTER_AREA));
    }
    //---

    // Solve optical flow from the coarsest scale to finest
    for (int iScale = nScales-1; iScale >= 0; --iScale)
    {
        if (iScale != nScales-1)
        {
            int nHeightNext = scaledOrigPred[iScale].r.rows;
            //double upscalingFactor = nHeightNext/(double)os.c.rows;

            if (!isAllNormal(os.c))
            {
                cout << "before resize to height " << nHeightNext << endl;
                throw 1;
            }

            cv::resize(os.c, os.c, Size{nLines,nHeightNext}, 0, 0, INTER_LINEAR);

            if (!isAllNormal(os.c))
            {
                cout << "after resize to height " << nHeightNext << endl;
                throw 1;
            }

            if (!isAllNormal(gammas[iScale]))
            {
                cout << "gammas " << nHeightNext << endl;
                throw 1;
            }

        }

        PRINT_NAMED(scaledOrigPred[iScale].r.rows);

        os = solveCombinationOnSingleScale(
                     scaledOrigPred[iScale],
                     scaledDesired[iScale],
                     os,
                     alphas[iScale],
					 aprioriOutput,
					 aprioriDeviationReward,
                     gammas[iScale],
                     rho,
                     SORomega,
                     nSORiterations);
    }
    //---

    //cv::resize(os.c, os.c, Size{nLines,nFrames}, 0, 0, INTER_LINEAR);

    Mat lineFlowEstimates = cvxret::backwardDerivative(os.c);
    Mat regionCountEstimates =
            aggregateLinesToRegionsInbetween(os.c) + cv::repeat(os.a, nFrames, 1);
    return {regionCountEstimates, lineFlowEstimates};
}


