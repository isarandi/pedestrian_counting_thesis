#include <CrowdCounting/RegionCounting/CountingFrame.hpp>
#include <CrowdCounting/CrowdCountingUtils.hpp>

#include <cvextra/colors.hpp>
#include <cvextra/cvret.hpp>
#include <cvextra/filesystem.hpp>
#include <cvextra/improc.hpp>
#include <cvextra/math.hpp>
#include <cvextra/strings.hpp>
#include <cvextra/visualize.hpp>

#include <opencv2/core/core.hpp>
#include <opencv2/core/mat.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <boost/filesystem.hpp>

#include <iomanip>
#include <iostream>
#include <fstream>
#include <string>
#include <vector>

using namespace std;
using namespace cv;
using namespace crowd;

void standardizeFileNames(string const& folder)
{
    vector<string> filePaths = cvx::filesystem::getFilePathsInFolder(folder);
    for (int i=0; i<filePaths.size(); ++i)
    {
        string extension = boost::filesystem::extension(filePaths[i]);
        string targetPath =
                folder + "/frame_" + cvx::str::zeropad(i,6) + extension;
        boost::filesystem::rename(filePaths[i], targetPath);
    }
}

void rescalePeoplePos(string filePath)
{
    vector<vector<Point2d> > sourcePositions = crowd::readPeoplePositions(filePath);

    ofstream outfile(filePath+"_relative");

    for (int iFrame = 0; iFrame < sourcePositions.size(); ++iFrame)
    {
        outfile << "Frame " << (iFrame+1) << endl;
        for (auto& p : sourcePositions[iFrame])
        {
            outfile << setprecision(8)
                    << cvx::math::clamp(p.x, 0.0, 0.999) << ", "
                    << cvx::math::clamp(p.y, 0.0, 0.999) << endl;
        }

        outfile << "--------" << endl << endl;
    }
}

void saveVariants(vector<CountingFrame> const& frames)
{
    Mat disk = cv::getStructuringElement(MORPH_ELLIPSE, Size(3,3));

    #pragma omp parallel for
    for (int iFrame = 0; iFrame < frames.size(); ++iFrame)
    {
        CountingFrame const& cf = frames[iFrame];

        Mat mask = cvxret::removeSmallConnectedComponents(cf.getMask(), 50);
        mask = cvxret::fillSmallHoles(mask, 20);
        Mat dilated = cvret::dilate(mask, disk);
        Mat outline = dilated-mask;
        cf.saveVariant("outlines", outline);

        Mat frame = cf.getFrame();
        Mat grayFrame = cvret::cvtColor(frame, COLOR_BGR2GRAY);
        Mat edges = cvret::Canny(grayFrame, 80, 150);
        cf.saveVariant("edges", edges);

        {
            Mat edgeSizedMask = cvxret::resizeBest(mask, edges.size());
            cv::threshold(edgeSizedMask, edgeSizedMask, 240, 255, THRESH_BINARY);
            Mat maskedEdges = cv::min(edges, edgeSizedMask);

            Mat edgesIllustration = cvret::cvtColor(edges, COLOR_GRAY2BGR);
            edgesIllustration.setTo(cvx::RED, maskedEdges);
            cf.saveVariant("masked_edges", edgesIllustration);
        }

        Mat fullFrame = cf.getFullResolutionFrame();
        cvx::resizeBest(mask, mask, fullFrame.size());
        cv::threshold(mask,mask, 240, 255, THRESH_BINARY);
        Mat illustration = cvx::visu::maskIllustration(fullFrame, mask);
        cf.saveVariant("mask_illustrations", illustration);
    }
}

void prepareDataset()
{
    //rescalePeoplePos("/work/sarandi/crowd/datasets/park/people_positions_relative.txt");
    standardizeFileNames("/work/sarandi/crowd/datasets/mall/frames");

    //vector<string> filenames = cvx::filesystem::getFilePathsInFolder("/work/sarandi/crowd/datasets/mall/frames");

//    Mat background =
//            cvx::resizeBest(
//                cv::imread("/work/sarandi/crowd/datasets/mall/mall_background.png", IMREAD_GRAYSCALE),
//                Size(320,240));

//    for (string &s : filenames)
//    {
//        Mat frame = cvx::resizeBest(
//                    cv::imread(s, IMREAD_GRAYSCALE), Size(320,240));
//        Mat mask = cvret::threshold(cvret::absdiff(frame, background), 40, 255, THRESH_BINARY) -
//                cvret::threshold(frame, 200, 255, THRESH_BINARY);


//        cvx::imwrite_force(
//                    "/work/sarandi/crowd/datasets/mall/masks/"+
//                    boost::filesystem::path(s)
//                    .filename()
//                    .replace_extension(".png")
//                    .string(),
//                    mask);
//    }



}

