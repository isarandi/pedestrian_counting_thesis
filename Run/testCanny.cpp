#include <boost/filesystem.hpp>
#include <cvextra/cvret.hpp>
#include <cvextra/improc.hpp>
#include <cvextra/io.hpp>
#include <cvextra/gui.hpp>
#include <cvextra/ImageLoadingIterable.hpp>
#include <cvextra/strings.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <Run/config.hpp>
#include <string>
#include <vector>

using namespace std;
using namespace cv;
using namespace cvx;

void testCanny()
{
//    vector<string> datasetNames = {
////            "crange_densities/11",
////            "crange_densities/12",
////            "crange_densities/13",
////            "crange_densities/14",
////            "crange_densities/21",
////            "crange_densities/22",
////            "crange_densities/23",
////            "crange_densities/24",
////            "crange_densities/31",
////            "crange_densities/32",
////            "crange_densities/33",
////            "crange_densities/34",
////            "crange_densities/41",
////            "crange_densities/42",
////            "crange_densities/43",
////            "crange_densities/44",
////            "crange_ausschnitt1",
//    		"ucsd_vid"
//            };
//
//    //vidd
//    double gaussRadius = 0.24;
//    double threshold1 = 177.9;
//    double threshold2 = 145.8;
//    int minSize = 5.257;
//
//    for (auto datasetName : datasetNames)
//    {
//        int iFrame = 0;
//        for (auto image : config::frames(datasetName)))
//        {
//            Mat1b grayImage = cvret::cvtColor(image, COLOR_BGR2GRAY);
//            auto blurred =
//                    cvret::GaussianBlur(
//                            grayImage,
//                            {11,11},
//                            gaussRadius,
//                            gaussRadius);
//            auto canny = cvret::Canny(blurred, threshold1, threshold2);
//            auto filtered =
//                    cvxret::removeSmallConnectedComponents(
//                            canny,
//                            minSize,
//                            cvx::Connectivity::EIGHT);
//            string filename = cvx::str::format("%06d.png", iFrame);
//            cvx::imwrite(config::cannyPath(datasetName)/filename, filtered);
//
//            ++iFrame;
//        }
//    }

    Mat image = config::frames("ucsd_vidd1_orig")[100];
    Mat1b grayImage = cvret::cvtColor(image, COLOR_BGR2GRAY);
    cvx::resizeBest(grayImage, grayImage, grayImage.size());

    cv::imshow("Original", image);

    cvx::gui::startTweaking({
        {"Gauss", 0, 20, 3},
        {"t1", 0, 300, 80},
        {"t2", 0, 300, 40},
        {"minSize", 1, 100, 30}},
        [&](map<string,double> const& params)
        {
            auto filtered =
                    cvxret::cannyPlus(
                            grayImage,
                            params.at("Gauss"),
                            params.at("t1"),
                            params.at("t2"),
                            (int)params.at("minSize"));

            return filtered;
        });

}
