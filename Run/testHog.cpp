#include <cvextra/colors.hpp>
#include <cvextra/cvret.hpp>
#include <cvextra/improc.hpp>
#include <cvextra/mats.hpp>
#include <Run/config.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/core/core_c.h>
#include <opencv2/core/mat.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/core/types_c.h>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/objdetect/objdetect.hpp>
#include <cmath>
#include <iostream>
#include <string>
#include <vector>

using namespace std;
using namespace cv;
using namespace cvx;
using namespace crowd;

auto getHOGIllustration(
        Mat const& origImg,
        vector<float> const& descriptorValues,
        Size winSize,
        Size cellSize,
        int nGradientBins,
        int scaleFactor,
        double gradientStrengthVisualizationFactor) -> Mat
{
    Size nCells = winSize / cellSize;

    vector<Mat1d> binGradientStrengths;
    for (int bin : cvx::irange(nGradientBins))
    {
        binGradientStrengths.push_back(Mat1d{nCells, 0.0});
    }

    // note: overlapping blocks lead to multiple updates of this sum!
    // we therefore keep track how often a cell was updated,
    // to compute average gradient strengths
    Mat1d cellUpdateCounter{nCells, 0.0};

    // there is a new block on each cell (overlapping blocks!) but the last one
    Size nBlocks = nCells - Size{1,1};

    // compute gradient strengths per cell
    int descriptorDataIdx = 0;

    for (Point iBlock : cvx::colMajorPoints(nBlocks))
    {
        for (Point iCell : cvx::colMajorPoints(Rect{iBlock, Size{2,2}}))
        {
            for (int bin : cvx::irange(nGradientBins))
            {
                binGradientStrengths[bin](iCell) += descriptorValues[descriptorDataIdx++];
            }

            ++cellUpdateCounter(iCell);
        }
    }

    // compute average gradient strengths
    for (int bin : cvx::irange(nGradientBins))
    {
        cv::divide(binGradientStrengths[bin], cellUpdateCounter, binGradientStrengths[bin]);
    }

    cout << "descriptorDataIdx = " << descriptorDataIdx << endl;

    // dividing 180° into 9 bins, how large (in rad) is one bin?
    double binSizeInRadians = CV_PI/(double)nGradientBins;
    double maxVecLen = cellSize.width/2.0;

    Mat illust = cvxret::resizeByRatio(origImg, scaleFactor);

    for (Point iCell : cvx::points(nCells))
    {
        Point drawTopLeft = iCell * cellSize;
        Point drawBottomRight = drawTopLeft + cellSize;

        cv::rectangle(
                illust,
                cvx::point2i(drawTopLeft*scaleFactor),
                cvx::point2i(drawBottomRight*scaleFactor),
                cvx::GRAY);

        Point2d drawCenter = drawTopLeft + cellSize/2.0;

        for (int bin : cvx::irange(nGradientBins))
        {
            double gradientStrength = binGradientStrengths[bin](iCell);
            if (gradientStrength < 1e-5)
            {
                continue;
            }

            double binAngleInRadians = (bin+0.5) * binSizeInRadians;
            Vec2d dirVec{std::cos(binAngleInRadians), std::sin(binAngleInRadians)};

            Vec2d visualVec =
                    dirVec *
                    gradientStrength *
                    maxVecLen *
                    gradientStrengthVisualizationFactor;

            cv::line(
                    illust,
                    cvx::point2i((drawCenter - visualVec) * scaleFactor),
                    cvx::point2i((drawCenter + visualVec) * scaleFactor),
                    cvx::BLUE);
        }
    }

    return illust;
}


auto getHOGFeatures(Mat const& input, Size cellSize, int nGradientBins) -> vector<float>
{
    cout << input.size() << endl;
    cout << cellSize << endl;
    HOGDescriptor descriptor{input.size(), cellSize*2, cellSize, cellSize, nGradientBins};
    vector<float> result;
    descriptor.compute(input, result, input.size(), Size{0,0});
    return result;
}

void testHog()
{
    Mat frame = cvxret::resizeByRatio(config::frames("crange_ausschnitt1")[0], 0.25);

    //Mat frame = cvret::resize(cvx::imread(config::DATA_PATH/"ped.png"), Size{50,50});
    Mat grayFrame = cvret::cvtColor(frame, COLOR_BGR2GRAY);

    Mat input = grayFrame(Rect{50,60, 8*4,8*4});
    Size cellSize{4,4};
    int nGradientBins = 9;

    vector<float> hogFeatures = getHOGFeatures(input, cellSize, nGradientBins);

    cv::imshow(
            "HOG",
            getHOGIllustration(
                Mat1d{input.size(),0.0},
                hogFeatures,
                input.size(),
                cellSize,
                nGradientBins,
                4,4));

    cv::imshow(
            "Frame+HOG",
            getHOGIllustration(
                cvret::cvtColor(input, COLOR_GRAY2BGR),
                hogFeatures,
                input.size(),
                cellSize,
                nGradientBins,
                16,4));

    cv::imshow("Frame", cvxret::resizeBest(input, input.size()*4));
    cvx::waitKey(' ');
}

