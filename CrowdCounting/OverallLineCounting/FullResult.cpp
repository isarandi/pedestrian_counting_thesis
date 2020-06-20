#include <CrowdCounting/OverallLineCounting/FullResult.hpp>
#include <CrowdCounting/Combination/Combiner.hpp>
#include <Illustrate/plots.hpp>
#include <Python/Pyplot.hpp>
#include <Run/config.hpp>
#include <cvextra/cvret.hpp>
#include <cvextra/utils.hpp>
#include <cvextra/io.hpp>
#include <cvextra/LoopRange.hpp>
#include <cvextra/mats.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <boost/filesystem/path.hpp>
#include <boost/property_tree/ptree.hpp>

using namespace std;
using namespace cv;
using namespace cvx;
using namespace crowd;
using namespace crowd::linecounting;


auto FullResult::
linePlot(bool cumulative, bool show) const -> pyx::Pyplot
{
    int nLines = desiredLineFlow.cols;
    int nFrames = desiredLineFlow.rows;

    pyx::Pyplot plt;
    auto fig_axarr = plt.subplots({2,nLines}, "sharey='row',sharex=True,figsize=(8,4)");
    auto fig = fig_axarr[0];
    auto axarr = fig_axarr[1];

    auto c = [&](Mat const& m){return cumulative ? cvxret::cumsum(m) : m;};

    for (int iLine : cvx::irange(nLines))
    {
        auto fdesired = Mat(desiredLineFlow.col(iLine).clone()).reshape(1);
        auto fpredicted = Mat(predictedLineFlow.mean.col(iLine).clone()).reshape(1);

        if (!cumulative)
        {
            auto variances = Mat(predictedLineFlow.variance.col(iLine).clone()).reshape(1);
            cv::GaussianBlur(fdesired, fdesired, Size{1,15}, 5.0,5.0, BORDER_CONSTANT);

            axarr[0][iLine].call("plot", {c(fdesired.col(0)), "g"});
            crowd::plotWithStdevAx(axarr[0][iLine], c(fpredicted.col(0)), cvret::sqrt(c(variances.col(0))), "r");

            axarr[1][iLine].call("plot", {c(fdesired.col(1)), "g"});
            crowd::plotWithStdevAx(axarr[1][iLine], c(fpredicted.col(1)), cvret::sqrt(c(variances.col(1))), "r");

        } else {
            axarr[0][iLine].call("plot", {c(fdesired.col(0)), "g"});
            axarr[0][iLine].call("plot", {c(fpredicted.col(0)), "r"});

            axarr[1][iLine].call("plot", {c(fdesired.col(1)), "g"});
            axarr[1][iLine].call("plot", {c(fpredicted.col(1)), "r"});
        }
    }

    plt.subplots_adjust({},"left=0.02, bottom=0.03, right=0.98, top=0.98, wspace=0.05, hspace=0.08");

    if (show)
    {
        plt.showAndClose();
    }
    return plt;

}

auto FullResult::
regionPlot(bool uncertainty, bool show) const -> pyx::Pyplot
{
    int nRegions = desiredRegionCounts.cols;
    int nFrames = desiredLineFlow.rows;

    pyx::Pyplot plt;
    auto fig_axarr = plt.subplots({nRegions,1}, "sharex=True,sharey=True,figsize=(8,8)");
    auto fig = fig_axarr[0];
    auto axarr = fig_axarr[1];


    for (int iRegion : cvx::irange(nRegions))
    {
        axarr[iRegion].call("plot",{desiredRegionCounts.col(iRegion), "g"});

        if (uncertainty)
        {
            crowd::plotWithStdevAx(axarr[iRegion], predictedRegionCounts.mean.col(iRegion), cvret::sqrt(predictedRegionCounts.variance.col(iRegion)), "b");
        } else {
            axarr[iRegion].call("plot",{predictedRegionCounts.mean.col(iRegion), "r"});
        }
    }

    if (show)
    {
        plt.showAndClose();
    }
    return plt;

}

auto FullResult::
regionFromLinePlot(bool show) const -> pyx::Pyplot
{
    int nRegions = desiredRegionCounts.cols;
    int nFrames = desiredLineFlow.rows;

    pyx::Pyplot plt;
    auto fig_axarr = plt.subplots({nRegions,1}, "sharex=True,sharey=True,figsize=(8,8)");
    auto fig = fig_axarr[0];
    auto axarr = fig_axarr[1];

    auto lineBasedRegionCounts =
            cvxret::cumsum(
                    crowd::linecounting::aggregateLinesToRegionsInbetween(predictedLineFlow.mean));

    for (int iRegion : cvx::irange(nRegions))
    {
        axarr[iRegion].call("plot",{desiredRegionCounts.col(iRegion), "g"});
        axarr[iRegion].call("plot",{lineBasedRegionCounts.col(iRegion), "r"});
    }

    if (show)
    {
        plt.showAndClose();
    }
    return plt;

}

auto FullResult::
plot(bool cumulative, bool show) const -> pyx::Pyplot
{
	int nLines = desiredLineFlow.cols;
	int nFrames = desiredLineFlow.rows;

	pyx::Pyplot plt;
	auto fig_axarr = plt.subplots({4,nLines}, "sharey='row',sharex=True,figsize=(16,8)");
	auto fig = fig_axarr[0];
	auto axarr = fig_axarr[1];

    auto oldLineBasedRegionCounts =
            cvxret::cumsum(
                    crowd::linecounting::aggregateLinesToRegionsInbetween(predictedLineFlow.mean));

    if (!predictedRegionCounts.mean.empty())
    {
        oldLineBasedRegionCounts += cv::repeat(predictedRegionCounts.mean.row(0), nFrames, 1);
    }

    auto c = [&](Mat const& m){return cumulative ? cvxret::cumsum(m) : m;};

	for (int iLine : cvx::irange(nLines))
	{
		auto fdesired = Mat(desiredLineFlow.col(iLine).clone()).reshape(1);
		auto fpredicted = Mat(predictedLineFlow.mean.col(iLine).clone()).reshape(1);

		Mat fimproved;
        if (!improvedRegionCounts.empty())
        {
            fimproved = Mat(improvedLineFlow.col(iLine).clone()).reshape(1);
        }

		auto variances = Mat(predictedLineFlow.variance.col(iLine).clone()).reshape(1);
		cv::GaussianBlur(fdesired, fdesired, Size{1,15}, 5.0,5.0, BORDER_CONSTANT);

		axarr[1][iLine].call("plot", {c(fdesired.col(0)), "g"});
		crowd::plotWithStdevAx(axarr[1][iLine], c(fpredicted.col(0)), cvret::sqrt(c(variances.col(0))), "k");

        if (!improvedRegionCounts.empty())
        {
            axarr[1][iLine].call("plot", {c(fimproved.col(0)), "r"});
        }

		axarr[2][iLine].call("plot", {c(fdesired.col(1)), "g"});
		crowd::plotWithStdevAx(axarr[2][iLine], c(fpredicted.col(1)), cvret::sqrt(c(variances.col(1))), "k");

        if (!improvedRegionCounts.empty())
        {
            axarr[2][iLine].call("plot", {c(fimproved.col(1)), "r"});
        }

		axarr[3][iLine].call("plot", {c(fdesired.col(0))  -c(fdesired.col(1)), "g"});
		axarr[3][iLine].call("plot", {c(fpredicted.col(0))-c(fpredicted.col(1)), "k"});

        if (!improvedRegionCounts.empty())
        {
            axarr[3][iLine].call("plot", {c(fimproved.col(0)) -c(fimproved.col(1)), "r"});
        }

		if (iLine >= nLines-1)
			continue;

		axarr[0][iLine].call("plot",{oldLineBasedRegionCounts.col(iLine), "k"});
		axarr[0][iLine].call("plot",{desiredRegionCounts.col(iLine), "g"});
		crowd::plotWithStdevAx(axarr[0][iLine], predictedRegionCounts.mean.col(iLine), cvret::sqrt(predictedRegionCounts.variance.col(iLine)), "b");

		if (!improvedRegionCounts.empty())
		{
		    axarr[0][iLine].call("plot",{improvedRegionCounts.col(iLine).clone(), "r"});
		}
	}

	plt.subplots_adjust({},"left=0.02, bottom=0.03, right=0.98, top=0.98, wspace=0.05, hspace=0.08");

	if (show)
	{
		plt.showAndClose();
	}
	return plt;

}

static
void takeOwnership(
		InputOutputArray requester,
		InputArray requests,
		InputArray ownables,
		OutputArray myShare)
{
	cv::divide(requester, requests, myShare);
	Mat reshapedMyShare = myShare.getMat().reshape(1);
	std::replace_if(
			reshapedMyShare.begin<double>(),
			reshapedMyShare.end<double>(),
			[](double d){return !(d>=0 && d <= 1);},
			0);
	cv::subtract(requester, myShare.getMat().mul(ownables), requester);

	Mat reshapedloneDesired = requester.getMat().reshape(1);
	std::replace_if(
			reshapedloneDesired.begin<double>(),
			reshapedloneDesired.end<double>(),
			[](double d){return !(d>=0);},
			0);
}

auto FullResult::
associate(cv::Mat2d const& predicted, int maxDistance) const -> AssociationResult
{
	int nFrames = desiredLineFlow.rows;
	if (maxDistance == 0)
	{
		maxDistance = nFrames-1;
	}

	Mat2d lonePredictions = predicted.clone();
	Mat2d loneDesired = desiredLineFlow.clone();
	Mat2d loneLookers = Mat2d::zeros(loneDesired.size());

	Mat2d minimum = cv::min(lonePredictions, loneDesired);
	Mat2d ownable = cv::max(minimum, 0);
	Mat2d negatives = cv::min(minimum, 0);

	lonePredictions -= ownable;
	loneDesired -= ownable;

	lonePredictions -= negatives;
	loneLookers -= negatives;

	Mat2d foundFlowForDistance{maxDistance+1, desiredLineFlow.cols, Vec2d{0,0}};
	Mat2d loneLookersForDistance{maxDistance+1, desiredLineFlow.cols, Vec2d{0,0}};

	cv::reduce(desiredLineFlow - loneDesired, foundFlowForDistance.row(0), 0, REDUCE_SUM);
	cv::reduce(loneLookers, loneLookersForDistance.row(0), 0, REDUCE_SUM);

	Mat2d desiredRequests{loneDesired.size(),Vec2d{0,0}};
	Mat2d lookerRequests{loneLookers.size(),Vec2d{0,0}};

	for (int distance : cvx::irange(1,maxDistance+1))
	{
        desiredRequests.setTo(0);
        desiredRequests.rowRange(distance, nFrames) += loneDesired.rowRange(0, nFrames-distance);
        desiredRequests.rowRange(0, nFrames-distance) += loneDesired.rowRange(distance, nFrames);


        lookerRequests.setTo(0);
        lookerRequests.rowRange(distance, nFrames) += loneLookers.rowRange(0, nFrames-distance);
        lookerRequests.rowRange(0, nFrames-distance) += loneLookers.rowRange(distance, nFrames);

		Mat2d lookerOwnables = cv::min(lonePredictions, lookerRequests);
		lonePredictions -= lookerOwnables;

		Mat2d desiredOwnables = cv::min(lonePredictions, desiredRequests);
		lonePredictions -= desiredOwnables;

		Mat myShare;
		takeOwnership(loneLookers.rowRange(distance, nFrames), lookerRequests.rowRange(0, nFrames-distance), lookerOwnables.rowRange(0, nFrames-distance), myShare);
        takeOwnership(loneLookers.rowRange(0, nFrames-distance), lookerRequests.rowRange(distance, nFrames), lookerOwnables.rowRange(distance, nFrames), myShare);

        takeOwnership(loneDesired.rowRange(distance, nFrames), desiredRequests.rowRange(0, nFrames-distance), desiredOwnables.rowRange(0, nFrames-distance), myShare);
        takeOwnership(loneDesired.rowRange(0, nFrames-distance), desiredRequests.rowRange(distance, nFrames), desiredOwnables.rowRange(distance, nFrames), myShare);

		cv::reduce(desiredLineFlow - loneDesired, foundFlowForDistance.row(distance), 0, REDUCE_SUM);
		cv::reduce(loneLookers, loneLookersForDistance.row(distance), 0, REDUCE_SUM);
	}

//    Mat2d precisionForDistance =
//            cvret::divide(foundFlowForDistance, cv::repeat(cvret::reduce(predictedLineFlow.mean, 0, REDUCE_SUM), nFrames, 1)+loneLookersForDistance);
//	Mat2d recallForDistance =
//	        cvret::divide(foundFlowForDistance, cv::repeat(cvret::reduce(desiredLineFlow, 0, REDUCE_SUM), nFrames, 1));

//
//	for (int col : cvx::irange(recallForDistance.cols*2))
//	{
//		pyx::Pyplot plt;
//		plt.plot(recallForDistance.reshape(1).col(col), "b");
//		plt.plot(precisionForDistance.reshape(1).col(col), "r");
//		plt.show();
//		plt.close();
//	}

	return {foundFlowForDistance, loneLookersForDistance};

}

auto FullResult::
confusion(
		Mat2d const& predicted,
		int distance
		) const -> Confusion
{
    auto assoc = associate(predicted, distance);

	double truePositiveTotal =
			cvret::reduce(
			        assoc.foundFlows.row(distance).reshape(1),
					1,
					REDUCE_SUM).at<double>(0,0);

	double desiredTotal = cv::sum(desiredLineFlow.reshape(1))[0];
	double predictedTotal = cv::sum(predicted.reshape(1))[0] + cv::sum(assoc.loneLookers.row(distance).reshape(1))[0];

	return Confusion{truePositiveTotal, predictedTotal, desiredTotal};
}

auto FullResult::
precisionRecallCurve(
        Mat2d const& predicted,
        int maxDistance
        ) const -> Mat1d
{
    auto assoc = associate(predicted, maxDistance);

    Mat1d truePositiveTotal =
            cvret::reduce(assoc.foundFlows.reshape(1), 1, REDUCE_SUM);

    double desiredTotal = cv::sum(desiredLineFlow.reshape(1))[0];

    Mat1d predictedTotal =
            cv::sum(predicted.reshape(1))[0] +
            cvret::reduce(assoc.loneLookers.reshape(1), 1, REDUCE_SUM);

    Mat1d averageTotal = (desiredTotal+predictedTotal)/2;

    return cvxret::hconcatAll({truePositiveTotal/predictedTotal, truePositiveTotal/desiredTotal, truePositiveTotal/averageTotal});
}

auto FullResult::
boxEvaluationCurve(cv::Mat2d const& predicted, bool averaged) const -> Mat2d
{
    Mat2d means{predicted.rows-1, predicted.cols};
    Mat2d cumulativeDifference = cvxret::cumsum(predicted-desiredLineFlow);

    for (int windowSize : cvx::irange(1,predicted.rows))
    {
        Mat2d absWindowErrors = cv::abs(
                cumulativeDifference.rowRange(windowSize, cumulativeDifference.rows) -
                cumulativeDifference.rowRange(0, cumulativeDifference.rows-windowSize));

        Mat meandst = means.row(windowSize-1);
        cv::reduce(absWindowErrors, meandst, 0, REDUCE_AVG);
    }

    return cvret::reduce(means, 1, REDUCE_AVG);
}

auto FullResult::
lineFlowUncertaintyQuality() const -> double
{
    return cvx::math::pearsonCorrelation(
            cvret::sqrt(predictedLineFlow.variance),
            cv::abs(predictedLineFlow.mean-desiredLineFlow));
}

auto FullResult::
regionUncertaintyQuality() const -> double
{
    return cvx::math::pearsonCorrelation(
            cvret::sqrt(predictedRegionCounts.variance),
            predictedRegionCounts.mean-desiredRegionCounts);
}

auto FullResult::
meanFinalAbsError(cv::Mat2d const& predicted) const -> double
{
    auto predictedTotals = cvret::reduce(predicted.reshape(1), 0, REDUCE_SUM);
    auto desiredTotals = cvret::reduce(desiredLineFlow.reshape(1), 0, REDUCE_SUM);

    return cv::mean(cvret::absdiff(predictedTotals, desiredTotals))[0];
}

auto FullResult::
meanFinalAbsRelError(cv::Mat2d const& predicted) const -> double
{
    auto predictedTotals = cvret::reduce(predicted.reshape(1), 0, REDUCE_SUM);
    auto desiredTotals = cvret::reduce(desiredLineFlow.reshape(1), 0, REDUCE_SUM);

    return cv::mean(cvret::divide(cvret::absdiff(predictedTotals, desiredTotals), desiredTotals), desiredTotals!=0)[0];
}

void FullResult::
horizontalAdd(FullResult const& r)
{
    cvx::hconcat(predictedLineFlow.mean, r.predictedLineFlow.mean, predictedLineFlow.mean);
    cvx::hconcat(predictedLineFlow.variance, r.predictedLineFlow.variance, predictedLineFlow.variance);
    cvx::hconcat(predictedRegionCounts.mean, r.predictedRegionCounts.mean, predictedRegionCounts.mean);
    cvx::hconcat(predictedRegionCounts.variance, r.predictedRegionCounts.variance, predictedRegionCounts.variance);
    cvx::hconcat(desiredLineFlow, r.desiredLineFlow, desiredLineFlow);
    cvx::hconcat(desiredRegionCounts, r.desiredRegionCounts, desiredRegionCounts);
    cvx::hconcat(improvedLineFlow, r.improvedLineFlow, improvedLineFlow);
    cvx::hconcat(improvedRegionCounts, r.improvedRegionCounts, improvedRegionCounts);
}

auto FullResult::
errorCharacteristics(
		cv::Mat2d const& predicted,
		int windowSize
		) const -> ErrorCharacteristics
{
    Mat2d cumulativeDifference = cvxret::cumsum(predicted-desiredLineFlow);

    SlidingWindow<int> window{windowSize, windowSize};
    int nWindows = window.sectionCountOver(predicted.rows);

    Mat2d windowErrors{nWindows, predicted.cols};
    for (int iWindow : cvx::irange(nWindows))
    {
    	cv::Range windowRange = cv::Range(window.ith(iWindow));

    	Mat dst = windowErrors.row(iWindow);
    	cv::subtract(
    	        cumulativeDifference.row(windowRange.end-1),
    	        cumulativeDifference.row(windowRange.start),
    	        dst);
    }

    Mat1d flatWindowErrors = cvx::reshapeCols(windowErrors, 1, 1);
    double meanOfErrors = cv::mean(flatWindowErrors)[0];
    double varianceOfErrors =
    		cvxret::variance(
    				flatWindowErrors, 0, SRC_TYPE,
					cvx::mats::matFromRows({{meanOfErrors}})).at<double>(0,0)
			* (flatWindowErrors.rows / (double)(flatWindowErrors.rows+1));

    return {meanOfErrors/windowSize, varianceOfErrors/windowSize};
}

auto FullResult::
load(cvx::bpath const& path) -> FullResult
{
    return FullResult{
        Mat1d(cvx::io::readDoubleMatFromCSV(path/"desiredRegionCounts.csv")),
        Mat2d(cvx::io::readDoubleMatFromCSV(path/"desiredLineFlow.csv").reshape(2)),
        PredictionWithConfidence{
                Mat1d(cvx::io::readDoubleMatFromCSV(path/"predictedRegionCounts.mean.csv")),
                Mat1d(cvx::io::readDoubleMatFromCSV(path/"predictedRegionCounts.variance.csv"))
        },
        PredictionWithConfidence2{
                Mat2d(cvx::io::readDoubleMatFromCSV(path/"predictedLineFlow.mean.csv").reshape(2)),
                Mat2d(cvx::io::readDoubleMatFromCSV(path/"predictedLineFlow.variance.csv").reshape(2))
        },
        Mat1d(cvx::io::readDoubleMatFromCSV(path/"improvedRegionCounts.csv")),
        Mat2d(cvx::io::readDoubleMatFromCSV(path/"improvedLineFlow.csv").reshape(2))
    };
}

void FullResult::
save(cvx::bpath const& path) const
{
    cvx::io::writeToCSV(path/"predictedLineFlow.mean.csv", predictedLineFlow.mean);
    cvx::io::writeToCSV(path/"predictedLineFlow.variance.csv", predictedLineFlow.variance);
    cvx::io::writeToCSV(path/"predictedRegionCounts.mean.csv", predictedRegionCounts.mean);
    cvx::io::writeToCSV(path/"predictedRegionCounts.variance.csv", predictedRegionCounts.variance);
    cvx::io::writeToCSV(path/"desiredLineFlow.csv", desiredLineFlow);
    cvx::io::writeToCSV(path/"desiredRegionCounts.csv", desiredRegionCounts);
    cvx::io::writeToCSV(path/"improvedLineFlow.csv", improvedLineFlow);
    cvx::io::writeToCSV(path/"improvedRegionCounts.csv", improvedRegionCounts);
}

auto FullResult::
describe() const -> boost::property_tree::ptree
{
	boost::filesystem::unique_path("%%%");
	if (folder.empty())
	{
		folder = config::RESULT_PATH/boost::filesystem::unique_path(cvx::timestamp()+"_%%%");
	}
	save(folder);

	boost::property_tree::ptree pt;
	pt.put("folder", folder.string());

	return pt;
}

auto FullResult::
create(boost::property_tree::ptree const& pt) -> std::unique_ptr<FullResult>
{
	bpath folder = pt.get<string>("folder");
	return stdx::make_unique<FullResult>(FullResult::load(folder));
}

auto ErrorCharacteristics::
getExpectedAbsError(double time) -> double
{
	double v = time*variance;
	double m = time*mean;
	return std::sqrt(2.0*v/CV_PI) * std::exp(-cvx::sq(m)/(2*v)) - m * std::erf(-m/(std::sqrt(2*v)));
}

auto ErrorCharacteristics::
getExpectedAbsErrorCurve(int size) -> Mat1d
{
	Mat1d result{size, 1};
	for (int i : cvx::irange(size))
	{
		result(i) = getExpectedAbsError(i+1);
	}
	return result;
}

auto FullResult::
meanRegionAbsError(
        cv::Mat1d const& predicted) const -> double
{
    return cv::mean(cv::abs(predicted-desiredRegionCounts))[0];
}

auto FullResult::
meanRegionAbsRelError(
        cv::Mat1d const& predicted) const -> double
{
    return cv::mean(cv::abs(cvret::divide(predicted-desiredRegionCounts, desiredRegionCounts)), desiredRegionCounts!=0)[0];
}



auto FullResult::
horizontalMerge(std::vector<FullResult> const& results) -> FullResult
{
    FullResult aggregateResult;
    for (auto const& result : results)
    {
        aggregateResult.horizontalAdd(result);
    }
    return aggregateResult;
}
