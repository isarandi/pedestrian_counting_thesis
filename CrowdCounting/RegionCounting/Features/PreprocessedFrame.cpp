#include <cvextra/improc.hpp>
#include <cvextra/cvret.hpp>
#include <cvextra/mats.hpp>
#include <cvextra/gui.hpp>
#include <CrowdCounting/RegionCounting/CountingFrame.hpp>
#include <CrowdCounting/RegionCounting/Features/PreprocessedFrame.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>

using namespace std;
using namespace cv;
using namespace cvx;
using namespace crowd;

PreprocessedFrame::PreprocessedFrame(CountingFrame const& countingFrame)
    : colorFrame(countingFrame.getFrame())
    , mask(countingFrame.getMask())
    , scaleMap(countingFrame.getScaleMap())
	, textonMap(countingFrame.getTextonMap())
{
//    Mat s = cv::getStructuringElement(MORPH_ELLIPSE, Size(3,3), Point(1,1));
//    cv::morphologyEx(mask, mask, MORPH_CLOSE, s);
//    cv::morphologyEx(mask, mask, MORPH_OPEN, s);

    mask = cvxret::removeSmallConnectedComponents(mask, 5);
    mask = cvxret::fillSmallHoles(mask, 10);

    cv::cvtColor(colorFrame, grayFrame, COLOR_BGR2GRAY);

//    cvx::gui::startTweaking({
//        {"Gauss", 0, 2, 0.74},
//        {"T1", 50, 150, 96.6},
//        {"T2", 50, 150, 81.3},
//        {"minSize", 0, 100, 24}},
//        [&](map<string,double> const& p)
//        {
//            return cvxret::vconcat(
//                    colorFrame,
//                    cvret::cvtColor(
//                            cvxret::cannyPlus(
//                                    grayFrame,
//                                    p.at("Gauss"),
//                                    p.at("T1"),
//                                    p.at("T2"),
//                                    p.at("minSize")), COLOR_GRAY2BGR));
//        }
//    );

    edges = cvxret::cannyPlus(grayFrame, 0.822, 82.8, 113.8, 20);


//    cv::imshow("", edges);
//    cv::waitKey();

    //edges = cvxret::Canny(grayFrame, edges, 80, 150);
//    cv::imshow("mask", mask);
//    cv::imshow("colorFrame", colorFrame);
//    cv::imshow("edges", edges);
//    cv::waitKey();
}
