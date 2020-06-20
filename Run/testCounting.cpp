#include <cvextra/gui.hpp>
#include <cvextra/strings.hpp>
#include <CrowdCounting/RegionCounting/AllToAllRegionCounter.hpp>
#include <CrowdCounting/RegionCounting/CountingTestCase.hpp>
#include <CrowdCounting/RegionCounting/CountingTestResult.hpp>
#include <CrowdCounting/RegionCounting/Features/Extractors/Area.hpp>
#include <CrowdCounting/RegionCounting/Features/Extractors/LocalBinaryPatterns.hpp>
#include <CrowdCounting/RegionCounting/Features/MultiFeatureExtractor.hpp>
#include <CrowdCounting/RegionCounting/FrameCollection.hpp>
#include <MachineLearning/KernelRidge.hpp>
#include <MachineLearning/Ridge.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <Run/setupDatasets.hpp>
#include <cmath>
#include <iostream>
#include <map>
#include <memory>
#include <string>
#include <vector>

using namespace std;
using namespace cv;
using namespace cvx;
using namespace crowd;

static void doGui()
{
//    auto extr = MultiFeatureExtractor{
//        Area(),
//        //Perimeter(),
//        //PerimeterOriHistogram(6),
//        //PerimeterToAreaRatio(),
//        //EdgeMinkowski(15),
//        //StatisticalLandscape(31),
//        //GrayLevelCooccurrence(),
////        FilterBank::LM(49),
//        LocalBinaryPatterns(1,8,15,Size(1,1),true),
//    };
//
//    auto datasets = crowd::getDatasets();
//    auto training = datasets["mall_kde"]->rangeFrom(5);
//    auto test = *datasets["PETS"];
//
//    double C = 10;
//    double gamma = 1;
//    double p = 1;
//    auto update = [&](){
//
//        cout << cvx::str::format("C=%.3e gamma=%.3e p=%.3e", C, gamma, p) << endl;
//
//        auto blobbased =
//                KalmanFilteredCrowdCounter(BlobBasedCounter(
//                    extr,
//                    NormalizedRegression(
//                        MultivariateAdapter(
//                            SupportVectorRegression(C, gamma, p)))), 0.7, true, true);
//
//        auto testCase = CountingTestCase{training, test, blobbased};
//        auto result = testCase.run();
//        cout << cvx::str::format("C=%.3e gamma=%.3e p=%.3e MSE=%.5f", C, gamma, p, result.getMeanSquaredError()) << endl;
//        auto figure = result.figure();
//        figure->show();
//        //auto plot = figure->(Size(1500, 800));
//
//        //cv::imshow("Counting", plot);
//    };
//
//    cv::namedWindow("Counting");
//    cvx::gui::createTrackbar(
//                "Counting", "C", std::log10(C), 0, 20,
//                [&](double val){C = std::pow(10, val); update();});
//
//    cvx::gui::createTrackbar(
//                "Counting", "Gamma", std::log10(gamma), -100, 100,
//                [&](double val){gamma = std::pow(10, val); update();});
//
//    cvx::gui::createTrackbar(
//                "Counting", "p", std::log10(p), -30, 0,
//                [&](double val){p = std::pow(10, val); update();});
//
//    cv::waitKey();

}

void runTests(vector<CountingTestCase*> testCases)
{
    auto results = std::vector<CountingTestResult>{};
    for (int iTestCase = 0; iTestCase < testCases.size(); ++iTestCase)
    {
        auto result = testCases[iTestCase]->run();
        result.saveToLog();
        result.saveMatlabCode();
        result.savePlot();

        results.push_back(result);
        delete testCases[iTestCase]; // free memory of extracted features etc.
    }

    for (int i=0; i<results.size(); ++i)
    {
        cout << results[i].getResultDescription() << endl;
    }
}

void testCounting()
{




//    doGui();

//    auto datasets = crowd::getDatasets();
//
//    // ======================= COUNTING MODELS ===========================
//
//    // ======== REGRESSION ========
//    auto optReg = SelfOptimizingRegression(
//                Ridge(1e-3),
//                0.7,
//                true,
//                pow(10,1/2.0));
//
//    auto optRegKernel = SelfOptimizingRegression(
//                KernelRidge(NoiseFreeRegularizedLeastSquaresParam{100}, 1e-6),
//                0.7,
//                true,
//                pow(10,1/2.0));
//
//    auto optSVM = SelfOptimizingRegression(
//                SupportVectorRegression(9, 1e-2, 1e-2),
//                0.7,
//                true,
//                pow(10,1/2.0));
//
//    auto normOptReg = NormalizedRegression(optReg);
//    auto normOptRegKernel = NormalizedRegression(optRegKernel);
//    auto normOptSVM = NormalizedRegression(optSVM);
//    auto attributeReg = CumulativeRegression(optReg, optReg, 0.5, false);
//    auto normAttributeReg = NormalizedRegression(attributeReg);
//
//
//    auto extr = MultiFeatureExtractor{
//        Area(),
//        //Perimeter(),
//        //PerimeterOriHistogram(6),
//        //PerimeterToAreaRatio(),
//        //EdgeMinkowski(15),
//        //StatisticalLandscape(31),
//        //GrayLevelCooccurrence(),
////        FilterBank::LM(49),
//        LocalBinaryPatterns(1,8,15,Size(1,1),true),
//    };
//
//    auto modelPerScale = ModelPerScaleCrowdCounter(
//                Size(8,8),
//                extr,
//                NormalizedRegression(Ridge(1e-3)));
//
//    auto blobbased = BlobBasedCounter(
//                extr,
//                NormalizedRegression(MultivariateAdapter(SupportVectorRegression(10, 10, 1))));
//
//    auto counter = AllToAllCrowdCounter(
//                Size(8,8),
//                MultiFeatureExtractor{ LocalBinaryPatterns(1,8,15,Size(1,1)) },
//                NormalizedRegression(Ridge(1e-3)));
//
//    // ======================= EVALUATION ===========================
//    auto testCases = std::vector<CountingTestCase*>{
//        new CountingTestCase(
//            datasets["mall_kde"]->rangeFrom(5),
//            *datasets["PETS"],
//            //datasets["mall_kde"]->range(5),
//            blobbased)
//    };
//
//    //Persistence::off();
//
//    runTests(testCases);
}



