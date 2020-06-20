#include "run.hpp"
#include "Run/Flow/runFlow.hpp"
#include "Run/LineCounting/runLineCounting.hpp"
#include "Run/LineCounting/runRegionCounting.hpp"
#include "Run/LineCounting/runCombination.hpp"
#include "Run/LineCounting/runFlowMosaicking.hpp"
#include "Run/config.hpp"
#include "Run/LineCounting/scenarios.hpp"
#include <CrowdCounting/LineCounting/FeatureSlices.hpp>
#include <CrowdCounting/LineCounting/LineCountingSet.hpp>
#include <CrowdCounting/LineCounting/LineLearningRepresenter.hpp>
#include <CrowdCounting/LineCounting/Features/Textons/TextonCreator.hpp>
#include <CrowdCounting/LineCounting/Features/Textons/TextureDescriptor.hpp>
#include <CrowdCounting/PersonLocations.hpp>
#include <Illustrate/illustrate.hpp>
#include <cvextra/LoopRange.hpp>
#include <cvextra/VideoIterable.hpp>
#include <cvextra/mats.hpp>
#include <cvextra/strings.hpp>
#include <cvextra/io.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/core/mat.hpp>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <functional>
#include <iostream>
#include <random>
#include <string>
#include <vector>

using namespace std;
using namespace cv;
using namespace cvx;
using namespace std::chrono;
using namespace crowd;
using namespace crowd::linecounting;

auto ff(Mat1d const& X, double rbfGamma) -> Mat1d
{
    Mat1d kernelMatrix{X.rows, X.rows};

    #pragma omp parallel for schedule(dynamic)
    for (int i=0; i<X.rows; ++i)
    {
        double const* x1 = reinterpret_cast<double const*>(X.ptr(i));
        double* ki = reinterpret_cast<double*>(kernelMatrix.ptr(i));

        for (int j=0; j<i; ++j)
        {
            double const* x2 = reinterpret_cast<double const*>(X.ptr(j));
            double sumOfSquaredDiffs = 0;

            for (int k = 0; k < X.cols; ++k)
            {
                sumOfSquaredDiffs += cvx::sq(x1[k]-x2[k]);
            }
            ki[j] = std::exp(-rbfGamma * sumOfSquaredDiffs);
            kernelMatrix(j,i) = ki[j];
        }

        ki[i] = 1.0;
    }

    return kernelMatrix;
}

void timeit()
{
    std::normal_distribution<double> distribution{0.0, 1.0f};
    std::mt19937 engine(245);
    auto generator = std::bind(distribution, engine);

    cv::Mat1d X{2000,2000};

    std::generate(X.begin(), X.end(), generator);

    double rbfGamma = std::exp(-2.8);
    vector<double> results;

    for (int i : cvx::irange(3))
    {
        auto t1 = high_resolution_clock::now();
        ff(X, rbfGamma);
        auto t2 = high_resolution_clock::now();
        auto duration = duration_cast<microseconds>(t2-t1).count();

        results.push_back(duration);
    }

    cout << cv::mean(results)[0] << endl;
}

void run()
{
//    LineCountingSet countingSet{"crange_ausschnitt1", LineSegment{{960/2.,0},{960/2.,540}}, {0,1500}};
//    auto slices = countingSet.loadSlices();
//
//    Mat1f flowX = cvret::extractChannel(slices.flow, 0);
//    Mat1f flowY = cvret::extractChannel(slices.flow, 1);
//    flowX = cvxret::medianFilter(flowX, Size{3,3});
//    cv::GaussianBlur(flowX, flowX, Size{5,5}, 0.02, 1.1);
//    cvx::setChannel(Mat1d{flowX}, slices.flow, 0);
//
//    slices.directionLabels = LineLearningRepresenter::segmentMovementStatic(cvret::merge({flowX,flowY}), slices.image, 1.0);
//    bpath p = "/home/sarandi/main/uni/crowd/writing/master_thesis/thesis_real/pictures";
//
//    cvx::imwrite(p/"flow_slice.png", cvx::visu::vectorFieldAsHSVAsBGR(slices.flow, 7).t());
//
//    Mat3b dirIllust = Mat3b::zeros(slices.directionLabels.size());
//    dirIllust.setTo(cvx::RED, slices.directionLabels==1);
//    dirIllust.setTo(cvx::GREEN, slices.directionLabels==2);
//    dirIllust.setTo(Scalar::all(225), slices.directionLabels==0);
//
//    Mat3b flowLeft = cvx::visu::vectorFieldAsHSVAsBGR(slices.flow, 7);
//    flowLeft.setTo(Scalar::all(255), slices.directionLabels!=1);
//
//    Mat3b flowRight = cvx::visu::vectorFieldAsHSVAsBGR(slices.flow, 7);
//    flowRight.setTo(Scalar::all(255), slices.directionLabels!=2);
//
//    cvx::imwrite(p/"flow_left_slice.png", flowLeft.t());
//    cvx::imwrite(p/"flow_right_slice.png", flowRight.t());
//
//
//
//    cvx::imwrite(p/"image_slice.png", slices.image.t());
//    cvx::imwrite(p/"foreground_slice.png", slices.foregroundMask.t());
//    cvx::imwrite(p/"direction_slice.png", dirIllust.t());
//    cvx::imwrite(p/"canny_slice.png", slices.canny.t());


    //crowd::run::testLineFlow();

//
//    auto seg = LineSegment{{720/2., 0.},{720/2., 320}};
//
//    cv::imshow("",
//            crowd::illust::createAnnotatedSlice(
//                    loc.getLineCrossings(seg),
//                    cvx::timeSlice(
//                            cvx::imagesIn(
//                                    config::framePath("ucsd_vidd1_orig_crop")).range(0, 1000),
//                                    seg),
//                    5).t());
//    cv::waitKey();

//    int iFrame = 0;
//    for (Mat3b const& im : cvx::imagesIn(config::framePath("ucsd_vidd1_orig")))
//    {
//        cvx::imwrite(config::framePath("ucsd_vidd1_orig_crop")/cvx::str::format("frame_%06d.jpg", iFrame), im.rowRange(160,480));
//        ++iFrame;
//    }

    //auto loc = PersonLocations::fromLineAnnotations("/work/sarandi/crowd/data/ground_truth/crange_long2.csv");
    //loc.writeToFile(config::locationPath("crange_long"));
    //crowd::run::testCombination();
    //crowd::run::testRegionCounting();
    crowd::run::testLineCounting();

    //crowd::run::testLineFlow();

    //testHog();
    //crowd::linecounting::runFlowMosaicking();

    //auto desc = LocalHOGDescriptor{Size{8,8}, 9};

//    Mat frame = config::frames("crange_long")[10];
//    cv::line(frame, Point{960/2,0}, Point{960/2,540}, cvx::YELLOW,4);
//    cvx::imwrite("/home/sarandi/main/uni/crowd/writing/master_thesis/talk2/figures/crange_single_line.png", frame);

    //testBackgroundSegmentation();

//    auto desc = FilterBankDescriptor{FilterBank::LM(25)};


//	TextonCreator tc{Size{960,540}/8, 8, desc, PerFeatureNormalizer{}, false};
//	tc.train(config::frames("crange_ausschnitt1").range(0,500), config::roiStencil("crange_ausschnitt1"));
}

