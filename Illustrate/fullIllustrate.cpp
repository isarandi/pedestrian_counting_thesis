#include <Illustrate/fullIllustrate.hpp>
#include <cvextra/colors.hpp>
#include <cvextra/coords.hpp>
#include <cvextra/core.hpp>
#include <cvextra/cvret.hpp>
#include <cvextra/filesystem.hpp>
#include <cvextra/improc.hpp>
#include <cvextra/io.hpp>
#include <cvextra/ImageLoadingIterable.hpp>
#include <cvextra/LoopRange.hpp>
#include <cvextra/math.hpp>
#include <cvextra/mats.hpp>
#include <cvextra/strings.hpp>
#include <cvextra/visualize.hpp>
#include <CrowdCounting/LineCounting/FeatureSlices.hpp>
#include <CrowdCounting/LineCounting/LineCountingSet.hpp>
#include <CrowdCounting/OverallLineCounting/FullResult.hpp>
#include <CrowdCounting/OverallLineCounting/OverallLineCounter.hpp>
#include <CrowdCounting/OverallLineCounting/SharedModelOverallLineCounter.hpp>
#include <CrowdCounting/PersonLocations.hpp>
#include <Illustrate/illustCrowdFlow.hpp>
#include <MachineLearning/Regression.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/core/mat.hpp>
#include <opencv2/core/operations.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <Run/config.hpp>
#include <Run/LineCounting/scenarios.hpp>
#include <string>
#include <vector>

using namespace std;
using namespace cv;
using namespace cvx;
using namespace crowd;
using namespace crowd::linecounting;

void crowd::fullIllustrate(
        OverallLineCountingScenario const& scenario,
        OverallLineCounter const& counter_,
        bpath const& path,
		double illustrationFactor)
{

    auto overallTest = scenario.tests[0];
    auto frames = config::frames(overallTest.datasetName).range(overallTest.frameRange);
    int nLines = counter_.getNumberOfLines();

    Size frameSize = frames[0].size();
    auto lines = crowd::getStandardLines(frameSize, nLines);

    auto counterPtr = counter_.clone();
    SharedModelOverallLineCounter& counter = dynamic_cast<SharedModelOverallLineCounter&>(*counterPtr);

    counter.train(scenario.trainings);


    FullResult results{counter.predict(overallTest), counter.getGroundTruth(overallTest)};

    Size illustSize = Size{
    	(int)(illustrationFactor*frameSize.width),
    	(int)(illustrationFactor*frameSize.height)};

    cv::VideoWriter videoWriter{
        path.string(),
        FourCC::XVID,
        5,
        Size{illustSize.width, illustSize.height+frameSize.height+5}};

    string datasetName = scenario.tests[0].datasetName;
    Range frameRange = scenario.tests[0].frameRange;

    PersonLocations loc = config::locations(datasetName).betweenFrames(frameRange);
    auto locByFrame = loc.getGroupedByFrame();

    LineCounter const& lineCounter = counter.getLineCounter();

    vector<FeatureSlices> lineSlices;
    vector<Mat3b> annotatedSlice;
    vector<LearningSet> testLearningSets;
    for (auto const& line : lines)
    {
        lineSlices.push_back(overallTest.getLineCountingSet(line).loadSlices());
        annotatedSlice.push_back(
                crowd::illust::createAnnotatedSlice(
                        loc.getLineCrossings(line),
                        lineSlices.back().image, 5));

        testLearningSets.push_back(
                lineCounter.createLearningSet(
                        overallTest.getLineCountingSet(line),
                        false));
    }

    Mat3b black{illustSize, cvx::V3B_BLACK};
    BinaryMat illustRoi =
    		cvxret::resizeBest(
				cvx::imread(
					config::roiStencilPath(
						scenario.tests[0].datasetName),
					IMREAD_GRAYSCALE),
				illustSize);

    Mat1d regionPred = results.predictedRegionCounts.mean;
    Mat1d regionDesired = results.desiredRegionCounts;

    int frameWindowSize = 20;
    int frameWindowStep = 5;

    for (int iFrame : cvx::irange(frames.size()))
    {
        Mat3b illust = cvxret::resizeBest(frames[iFrame], illustSize);
        Mat3b darkened = cvret::addWeighted(black, 0.5, illust, 0.5, 0.0);
        darkened.copyTo(illust, illustRoi==0);

        for (int iLine : cvx::irange(nLines))
        {
            auto seg = illustrationFactor*lines[iLine];
            Point p1 = cvx::point2i(seg.p1);
            cv::line(illust, p1, cvx::point2i(seg.p2), cvx::YELLOW, 2);

            Mat2d cumulValid = cvxret::cumsum(results.desiredLineFlow.col(iLine));
            cvx::putTextCairo(illust, cvx::str::format("%.2f",cumulValid(iFrame)[0]), seg.p1+cv::Vec2d{-25,+15}, "arial", 15, cvx::GREEN);
            cvx::putTextCairo(illust, cvx::str::format("%.2f",cumulValid(iFrame)[1]), seg.p1+cv::Vec2d{+25,+15}, "arial", 15, cvx::GREEN);

            Mat2d cumulPred = cvxret::cumsum(results.predictedLineFlow.mean.col(iLine));
            cvx::putTextCairo(illust, cvx::str::format("%.2f",cumulPred(iFrame)[0]), seg.p1+cv::Vec2d{-25,+30}, "arial", 15, cvx::RED);
            cvx::putTextCairo(illust, cvx::str::format("%.2f",cumulPred(iFrame)[1]), seg.p1+cv::Vec2d{+25,+30}, "arial", 15, cvx::RED);

//            Mat1d cumulImproved = cvxret::cumsum(Mat(improved.l.col(iLine).clone()).reshape(1));
//            cvx::putTextCairo(illust, cvx::str::format("%.2f",cumulImproved(iFrame)[0]), seg.p1+cv::Vec2d{-25,+45}, "arial", 15, cvx::ORANGE);
//            cvx::putTextCairo(illust, cvx::str::format("%.2f",cumulImproved(iFrame)[1]), seg.p1+cv::Vec2d{+25,+45}, "arial", 15, cvx::ORANGE);

            lineSlices[iLine];
            ++iLine;
        }

        for (int iRegion : cvx::irange(regionPred.cols))
        {
        	LineSegment prevLine =
        			(iRegion == 0) ?
        					LineSegment{{0,0},{0,frameSize.height-1}} :
							lines[iRegion-1];
			LineSegment nextLine =
					(iRegion == regionPred.cols-1) ?
							LineSegment{{frameSize.width-1,0},{frameSize.width-1,frameSize.height-1}} :
							lines[iRegion];
        	Point2d regionCenter = illustrationFactor*(prevLine.p1+prevLine.p2+nextLine.p1+nextLine.p2)/4.0;

            cvx::putTextCairo(
                    illust,
                    cvx::str::format("%.2f",regionDesired(iFrame,iRegion)),
					regionCenter,
                    "arial",
                    15,
                    cvx::GREEN);

            cvx::putTextCairo(
                    illust,
                    cvx::str::format("%.2f",regionPred(iFrame,iRegion)),
					regionCenter+cv::Vec2d{0,20},
                    "arial",
                    15,
                    cvx::RED);
        }

        for (PersonInstance& pi : locByFrame[iFrame])
        {
            if (cvx::contains(illust, pi.pos))
            {
                cv::circle(illust, cvx::point2i(pi.pos*illustrationFactor), 6, cvx::RED, cvx::FILL);
            }
        }

        Mat3b illust2{frameSize.height+5, illust.cols, cvx::V3B_BLACK};
        int iWindow = (iFrame-frameWindowSize/2)/frameWindowStep;

        for (int iLine : cvx::irange(nLines))
        {
            auto seg = lines[iLine];
            int x = (int)seg.p1.x;

            Rect roi{0, iWindow*frameWindowStep, seg.floorLength(), frameWindowSize};
            if (!cvx::contains(annotatedSlice[iLine], roi))
            {
                continue;
            }

            vector<Mat3b> sliceParts{
                    annotatedSlice[iLine](roi).t(),
                    lineSlices[iLine].canny(roi).t(),
                    cvret::cvtColor(lineSlices[iLine].foregroundMask(roi), COLOR_GRAY2BGR).t(),
                    cvx::visu::vectorFieldAsHSVAsBGR(lineSlices[iLine].flow(roi), 3).t()};

            int n = sliceParts.size();
            int d = 2;
            int tot = n*frameWindowSize+(n-1)*d;

            int iPart = 0;
            for (Mat3b const& slicePart : sliceParts)
            {
                int pos = x*illustrationFactor - tot/2+iPart*(frameWindowSize+d);

                Mat target = illust2(Rect{pos, 0, slicePart.cols, slicePart.rows});
                slicePart.copyTo(target);
                ++iPart;
            }
        }

        Mat1d jacobian = lineCounter.getRegression().getJacobian(testLearningSets[nLines/2].X.row(iWindow));
        pyx::Pyplot plt;
        plt.stem(jacobian);

        Mat plotImage = plt.renderAndClose();
        cv::imshow("l", plotImage);
        cv::waitKey();

        videoWriter << cvxret::vconcatAll({illust, illust2});

    }

}
