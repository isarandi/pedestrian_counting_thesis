#include <boost/filesystem.hpp>
#include <boost/iterator/iterator_facade.hpp>
#include <cvextra/coords.hpp>
#include <cvextra/core.hpp>
#include <cvextra/cvenums.hpp>
#include <cvextra/cvret.hpp>
#include <cvextra/filesystem.hpp>
#include <cvextra/gui.hpp>
#include <cvextra/improc.hpp>
#include <cvextra/io.hpp>
#include <cvextra/ImageLoadingIterable.hpp>
#include <cvextra/math.hpp>
#include <cvextra/mats.hpp>
#include <cvextra/visualize.hpp>
#include <CrowdCounting/PersonLocations.hpp>
#include <Flow/flowSlice.hpp>
#include <Flow/lineFlow.hpp>
#include <Illustrate/illustLineFlow.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/core/mat.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <Run/config.hpp>
#include "runFlow.hpp"
#include <iostream>
#include <map>
#include <string>
#include <vector>

using namespace std;
using namespace cv;
using namespace cvx;
using namespace crowd;
using namespace crowd::lineopticalflow;

//vidf
//alpha = 0.01747;
//beta = 0.01369;
//omega = 0.63112;
//nScales = 1.434;
//K = 13.773;
//L = 2.224;
//M = 4.104;

//alpha = 0.02665;
//beta = 0.03052;
//omega = 0.65943;
//nScales = 3.044;
//K = 8.913;
//L = 6.2;
//M = 3.552;


void crowd::run::tweakFlowParameters()
{
    auto images = config::frames("ucsd_vidd1_orig").range(0,500).load();

//    Rect roi{960/2, 0, 50, 540};
//
//    for (int iFrame : cvx::irange(170, 1000))
//    {
//        Mat3b im1 = images[iFrame](roi);
//        Mat3b im2 = images[iFrame+1](roi);
//
//        cout << iFrame << endl;
//
//        LineSegment segInRoi = {{roi.width/2.0, 0}, {roi.width/2.0, roi.height}}; //guiGetLineSegment(*images.begin());
//
//        OpticalFlowOptions options =
//                OpticalFlowOptions::fromFile(config::configPath("crange_ausschnitt1", "flow"));
//
//        Mat2d flow = crowd::lineopticalflow::createFlowSlice(
//            vector<Mat>{im1, im2},
//            segInRoi,
//            0, 0,
//            options);
//
//        Mat m = cvx::illust::inspectFlow(
//                flow, im1, im2,
//                segInRoi, 1, 5)[0];
//
//        cv::imshow("j", m);
//        cvx::imwrite("/home/sarandi/main/uni/crowd/writing/master_thesis/thesis_real/pictures/flow_illust4.png", m);
//        cv::waitKey();
//    }
//

    Size s = images[0].size();
    LineSegment seg = {{s.width/2, 0}, {s.width/2, s.height}}; //guiGetLineSegment(*images.begin());

    Mat3b imageSlice = cvx::timeSlice(images, seg).t();
    //LineSegment segInRoi{{12, 0}, {12, 50}};
    cvx::gui::startTweaking({
        {"alpha", 0.01, 0.1, 0.01585},
        {"beta", 0.01, 0.1, 0.04213},
        {"omega", 0.5, 1.99, 0.78906},
        {"nScales", 1, 8, 5},
        {"K", 3, 30, 3},
        {"L", 2, 10, 2},
        {"M", 2, 10, 2}},
        [&](map<string,double> const& params){
            OpticalFlowOptions options;
                    //OpticalFlowOptions::fromFile(config::configPath("crange_ausschnitt1", "flow"));;
            options.smoothnessAlpha = params.at("alpha");
            options.magnitudeBeta = params.at("beta");
            options.epsilonSquare = 1e-8;
            options.SORomega = params.at("omega");
            options.nScales = (int)params.at("nScales");
            options.nIterationsK = (int)params.at("K");
            options.nIterationsL = (int)params.at("L");
            options.nIterationsSOR = (int)params.at("M");

            Mat2d flow = crowd::lineopticalflow::createFlowSlice(
                images,
                seg, 0,0,
                options
            ).t();

            return cvxret::hconcat(imageSlice(cvx::fullRect(flow)), cvx::visu::vectorFieldAsHSVAsBGR(flow, 2));
            //auto inspect = cvx::illust::inspectFlow(flow, im1, im2, segInRoi, 100);
            //return cvxret::resizeByRatio(cvxret::hconcat(inspect[0], inspect[1]), 0.1);
        }, true);

}

void tweakSegmentation()
{
    //--- Load flow and feature slices

//    auto display =
//            cvx::gui::TweakableDisplay(false,
//                "flow segmentation",
//                {
//                        {"thresh", 0., 10., 1.5},
//                        {"gauss_t", 0.001, 40, 10},
//                        {"gauss_s", 0.001, 40, 1.5},
//                        {"median_range", 0, 5, 3}},
//                [&](map<string,double> const& params){
//                    Mat1f flowX = cvret::extractChannel(flowSlice, 0).clone();
//
//                    cv::medianBlur(flowX, flowX, ((int)params.at("median_range"))/2*2+1);
//                    cv::GaussianBlur(flowX, flowX, {31,7}, params.at("gauss_t"), params.at("gauss_s"));
//
//                    Mat1b result = Mat1b::zeros(flowSlice.size());
//                    result.setTo((int)CrossingDir::LEFTWARD, flowX < -params.at("thresh"));
//                    result.setTo((int)CrossingDir::RIGHTWARD, flowX > params.at("thresh"));
//
//                    Mat3b leftIllust = cvx::visu::maskIllustration(imageSlice, result==(int)CrossingDir::LEFTWARD);
//                    Mat3b rightIllust = cvx::visu::maskIllustration(imageSlice, result==(int)CrossingDir::RIGHTWARD);
//                    return cvxret::resizeByRatio(cvxret::hconcat(leftIllust, rightIllust).t(), 0.7);
//                }
//    );
//    display.show();
//    cv::waitKey();
}

void crowd::run::testLineFlow()
{
	tweakFlowParameters();
    //tweakSegmentation();
   //computeFlowSequence();
}
