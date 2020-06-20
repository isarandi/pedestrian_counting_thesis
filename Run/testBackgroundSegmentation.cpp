#include <boost/filesystem.hpp>
#include <BackgroundSegmentation/LikelihoodModel.hpp>
#include <BackgroundSegmentation/NeighborhoodBackgroundModel.hpp>
#include <BackgroundSegmentation/PixelFeatures/HSVCylinderFeatureExtractor.hpp>
#include <BackgroundSegmentation/PixelFeatures/IntensityAndGradientFeatureExtractor.hpp>
#include <cvextra/colors.hpp>
#include <cvextra/cvret.hpp>
#include <cvextra/filesystem.hpp>
#include <cvextra/gui.hpp>
#include <cvextra/improc.hpp>
#include <cvextra/io.hpp>
#include <cvextra/ImageLoadingIterable.hpp>
#include <cvextra/strings.hpp>
#include <cvextra/visualize.hpp>

#include <Python/EasyObject.hpp>
#include <Python/Pyplot.hpp>
#include <Python/PythonEnvironment.hpp>

#include <opencv2/core/core.hpp>
#include <opencv2/core/core_c.h>
#include <opencv2/core/mat.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/core/types_c.h>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <Run/config.hpp>
#include <cmath>
#include <iostream>
#include <string>
#include <vector>

using namespace std;
using namespace cv;
using namespace cvx;
using namespace crowd::bg;
using namespace crowd;
using namespace boost::filesystem;

template <typename ImageSequence>
static void visualize(
        ImageSequence& images,
        Size procSize,
        double threshold,
        double p,
        int displayFrame)
{
    auto pfe = HSVCylinderFeatureExtractor{0.7};

    auto bgs = NeighborhoodBackgroundModel{
                    threshold,
                    30,
                    threshold * 100,
                    pfe,
                    LikelihoodModel(p, 40),
                    0.1};

    auto showSize = Size{640,480};

    bgs.initialize(procSize);
    int iFrame = 0;

    for (auto const& original : images)
    {
        auto procOriginal = cvxret::resizeBest(original, procSize);

        BackgroundModel& bg = bgs;
        auto procMask = bg.segmentNextRet(procOriginal);

        if (iFrame++ == displayFrame)
        {
            auto showOriginal = cvxret::resizeBest(original, showSize);
            auto showMask = cvret::resize(procMask, showSize, 0, 0, INTER_NEAREST);

            auto illustration = cvx::visu::maskIllustration(showOriginal, showMask);
            cv::putText(illustration,
                        cvx::str::format("#%06d", iFrame),
                        Point{10,50},
                        FONT_HERSHEY_COMPLEX,
                        0.8,
                        cvx::WHITE,
                        1,
                        CV_AA);

            cv::imshow("BG", illustration);
            return;
        }
    }

}

void createMasks(
        path folderPath,
		path initPath,
        Size procSize,
        BackgroundModel& bgs,
        path maskOutputPath,
        path videoPath,
        bool display)
{

    if (!maskOutputPath.empty())
    {
        boost::filesystem::create_directories(maskOutputPath);
    }

//    Size showSize(640,480);
//
//    VideoWriter videoWriter;
//    if (!videoPath.empty())
//    {
//        videoWriter = VideoWriter(videoPath.string(), FourCC::XVID, 15, showSize);
//    }

    bgs.initializeWithImages(procSize, cvx::imagesIn(initPath));

    int iFrame = 0;
    for (Mat const& original : cvx::imagesIn(folderPath))
    {
        cvx::io::statusUpdate(cvx::str::format("Segmenting frame #%d (in %s)", iFrame, folderPath));

        Mat procOriginal = cvxret::resizeBest(original, procSize);
//        Mat showOriginal = cvxret::resizeBest(original, showSize);

        Mat procMask = bgs.segmentNextRet(procOriginal);
//        Mat showMask = cvret::resize(procMask, showSize, 0, 0, INTER_NEAREST);

//        Mat illustration = cvx::visu::maskIllustration(showOriginal, showMask);
//        cv::putText(illustration,
//                    cvx::str::format("#%06d", iFrame),
//                    Point(10,50),
//                    FONT_HERSHEY_COMPLEX,
//                    0.8,
//                    cvx::WHITE,
//                    1,
//                    CV_AA);

        if (!maskOutputPath.empty())
        {
            path maskFilePath = maskOutputPath /
                    cvx::str::format("frame_%06d.png", iFrame);
            cvx::imwrite(maskFilePath, procMask);
        }
//
//        if (!videoPath.empty())
//        {
//            videoWriter.write(illustration);
//        }
//
//        if (display)
//        {
//            cv::imshow("default", illustration);
//            cv::waitKey(10);
//        }

        ++iFrame;
    }
}

static void doGui()
{
	Size procSize{238,158};
	Size showSize{238*4,158*4};

    auto images = config::frames("ucsd_vidd");

    auto display =
        cvx::gui::TweakableDisplay(true,
            "Line counting",
            {
                    {"vscale", 0.2, 2, 0.7},
                    {"threshold", 0, 2000, 1000},
                    {"p", 0.001, 0.05, 0.01},
                    {"selectiveThreshold", 200, 2000, 1500},
					{"gradientScale", 0.0, 2.0, 0.25},
            },
            [&](map<string,double> const& params){

                auto extr = IntensityAndGradientFeatureExtractor{params.at("gradientScale")};
                double threshold = params.at("threshold");
                double selectiveThreshold = params.at("selectiveThreshold");
                double p = params.at("p");

                auto bgs = NeighborhoodBackgroundModel{
                                threshold,
                                30,
								selectiveThreshold,
                                extr,
                                LikelihoodModel(p, 40),
                                1.5};

                bgs.initializeWithImages(procSize, config::frames("ucsd_vidd_background_init"));

                int iFrame = 0;
                int displayFrame = 10;

                for (Mat const& original : images)
                {
                    auto procOriginal = cvxret::resizeBest(original, procSize, 0, 0);
                    auto procMask = bgs.segmentNextRet(procOriginal);

                    if (iFrame++ == displayFrame)
                    {
                        auto showOriginal = cvxret::resizeBest(original, showSize);
                        auto showMask = cvret::resize(procMask, showSize, 0, 0, INTER_NEAREST);

                        auto illustration = cvx::visu::maskIllustration(showOriginal, showMask);
                        cv::putText(illustration,
                                    cvx::str::format("#%06d", iFrame),
                                    Point{10,50},
                                    FONT_HERSHEY_COMPLEX,
                                    0.8,
                                    cvx::WHITE,
                                    1,
                                    CV_AA);

                        return illustration;
                    }
                }
            }
    );
    display.show();
    cv::waitKey();
}

void playVidf()
{
    double threshold = 576;
    double p = 0.001147;
    double selectiveThreshold = 1087.4;
    double gradientScale = 0.322;

    auto extr = IntensityAndGradientFeatureExtractor{gradientScale};

    auto bgs = NeighborhoodBackgroundModel{
                    threshold,
                    30,
                    threshold * 100,
                    extr,
                    LikelihoodModel(p, 40),
                    1.5};

    Size size{238,158};
    vector<string> datasetNames = {
            "ucsd_vidf"
            };

    for (auto const& datasetName : datasetNames)
    {
        createMasks(
                config::framePath(datasetName),
				config::framePath(datasetName+"_background_init"),
                size,
                bgs,
                (true ? config::maskPath(datasetName) : ""),
                (false ? config::maskPath(datasetName)/"masks.avi" : ""),
                false);
    }
}

void playVidd()
{
    double threshold = 272;
    double p = 0.001147;
    double selectiveThreshold = 1087.4;
    double gradientScale = 0.324;

    auto extr = IntensityAndGradientFeatureExtractor{gradientScale};

    auto bgs = NeighborhoodBackgroundModel{
                    threshold,
                    30,
                    threshold * 100,
                    extr,
                    LikelihoodModel(p, 40),
                    1.5};

    Size size{238,158};
    vector<string> datasetNames = {
            "ucsd_vidd"
            };

    for (auto const& datasetName : datasetNames)
    {
        createMasks(
                config::framePath(datasetName),
				config::framePath(datasetName+"_background_init"),
                size,
                bgs,
                (true ? config::maskPath(datasetName) : ""),
                (false ? config::maskPath(datasetName)/"masks.avi" : ""),
                false);
    }
}

void play()
{
    auto extr = HSVCylinderFeatureExtractor{0.7};

    double threshold = 1000;
    double p = 0.01;

    auto bgs = NeighborhoodBackgroundModel{
                    threshold,
                    30,
                    threshold * 100,
                    extr,
                    LikelihoodModel(p, 40),
                    1.5};

    //auto size = Size{480,270}/2;
    auto size = Size{720,320}/3;

    vector<string> datasetNames = {
//            "crange_densities/11",
//            "crange_densities/12",
//            "crange_densities/13",
//            "crange_densities/14",
//            "crange_densities/21",
//            "crange_densities/22",
//            "crange_densities/23",
//            "crange_densities/24",
//            "crange_densities/31",
//            "crange_densities/32",
//            "crange_densities/33",
//            "crange_densities/34",
//            "crange_densities/41",
//            "crange_densities/42",
//            "crange_densities/43",
//            "crange_densities/44",
            //"crange_long",
            "ucsd_vidd1_orig"
            };

    for (auto const& datasetName : datasetNames)
    {
        createMasks(
                config::framePath(datasetName),
				config::framePath("ucsd_vidd_orig_background_init"),
                size,
                bgs,
                (true ? config::maskPath(datasetName) : ""),
                (false ? config::maskPath(datasetName)/"masks.avi" : ""),
                false);
    }
}

void testBackgroundSegmentation()
{

	play();


//    auto featureExtractor = HSVCylinderFeatureExtractor{0.7};
//
//    double threshold = 1000;
//    double p = 0.01;
//
//    auto backgroundModel =
//            NeighborhoodBackgroundModel{
//                threshold,
//                30,
//                threshold * 100,
//                featureExtractor,
//                LikelihoodModel(p, 500),
//                1.5};
//
//    Size processingSize{480,270};
//
//    createMasks(
//                config::DATA_PATH/"frames/crange_ausschnitt1",
//                processingSize,
//                backgroundModel,
//                config::DATA_PATH/"intermediate/crange_ausschnitt1/masks",
//                config::DATA_PATH/"intermediate/crange_ausschnitt1/mask_video.avi",
//                false);
//
//
//    Point point{250,190};
//
//
//
//    bpath folderPath = config::DATA_PATH/"frames/crange_ausschnitt1";
//
//    vector<Point3d> colorSpacePoints;
//    vector<Scalar> colors;
//
//    auto images = cvx::imagesIn(folderPath);
//    auto sub = cvx::subiterable(images, 0, 1000);
//
//    int iFrame = 0;
//    for (auto const& original : sub)
//    {
//        //cvx::io::statusUpdate(cvx::str::format("Segmenting frame #%d (in %s)", iFrame, folderPath));
//        Mat3b procOriginal = cvxret::resizeBest(original, processingSize);
//
//        Vec3b rgb = procOriginal(point);
//        Vec3b hsv = cvx::cvtColor(rgb, COLOR_RGB2HSV);
//        double h = hsv[0] / 180.0;
//        double s = hsv[1] / 255.0;
//        double v = hsv[2] / 255.0;
//
//        double factor = s*(0.5-std::abs(0.5-v));
//        double x = std::cos(2*CV_PI*h) * factor;
//        double y = std::sin(2*CV_PI*h) * factor;
//
//        colorSpacePoints.push_back({x,y,v});
//        colors.push_back(cvx::toScalar(rgb));
//
//        iFrame++;
//    }
//
//    Plotter plotter;
//    plotter.scatter(colorSpacePoints, colors, "X", "Y", "V");
//    plotter.show();


////////////////////////////////////////////////////

//    omp_set_num_threads(1);

//    auto extr = HSVCylinderFeatureExtractor{0.7};

//    double threshold = 1000;
//    double p = 0.01;

//    auto bgs = NeighborhoodBackgroundModel{
//                    threshold,
//                    30,
//                    threshold * 100,
//                    extr,
//                    LikelihoodModel(p, 40),
//                    0.1};

//    auto size = Size{320,240};

//    createMasks("/work/sarandi/crowd/datasets/mall/frames", size, bgs, "", "", true);
}
