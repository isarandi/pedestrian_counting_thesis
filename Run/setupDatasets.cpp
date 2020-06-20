#include <cvextra/core.hpp>
#include <cvextra/cvenums.hpp>
#include <cvextra/improc.hpp>
#include <cvextra/LoopRange.hpp>
#include <cvextra/math.hpp>
#include <cvextra/vectors.hpp>
#include <Camera/PinholeCameraModel.hpp>
#include <Camera/RealisticCameraModel.hpp>
#include <CrowdCounting/RegionCounting/CountingFrame.hpp>
#include <CrowdCounting/RegionCounting/FrameCollection.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/core/mat.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include "setupDatasets.hpp"
#include <map>
#include <memory>
#include <string>
#include <vector>

using namespace std;
using namespace cv;
using namespace cvx;
using namespace crowd;

string parentFolder = "/work/sarandi/crowd/datasets/";

typedef map<string, shared_ptr<FrameCollection>> DatasetCollection;

void createMall(DatasetCollection& datasets);
void createPETS(DatasetCollection& datasets);
void createUCSD(DatasetCollection& datasets);

//Mat renderLinearScaleMap(double y1, double scale1, double y2, double scale2, Size size)
//{
//    Mat1d scaleMap(size);
//
//    for (Point p : cvx::points(size))
//    {
//        double yRel = p.y / static_cast<double>(size.height);
//        scaleMap(p) = cvx::math::linearRescale(yRel, y1, y2, scale1, scale2);
//    }
//    return scaleMap;
//}
//
//DatasetCollection crowd::getDatasets()
//{
//    auto datasets = DatasetCollection{};
//
//    createMall(datasets);
//    createPETS(datasets);
//    createUCSD(datasets);
//
//    return datasets;
//}
//
//void storeDataset(string name,
//                  vector<CountingFrame> const& frames,
//                  DatasetCollection& datasets)
//{
//    datasets[name] = std::make_shared<FrameCollection>(name, frames);
//}
//
//void createMall(DatasetCollection& datasets)
//{
//    PinholeCameraModel mallCam(8000, 0.09, 4082, Size(640,480));
//    BinaryMat roi(cv::imread(parentFolder+"/mall/roi.png", IMREAD_GRAYSCALE));
//    Size size(320,240);
//
//    cvx::resizeBest(roi, roi, size);
//    cv::threshold(roi, roi, 240, 255, THRESH_BINARY);
//
////    vector<CountingFrame> mallFrames_AbsDiffMask =
////            CountingFrame::allFromFolder(
////                parentFolder+"/mall/",
////                "masks_absdiff",
////                size,
////                mallCam.renderScaleMap(size)
////                );
//
//    auto mallFrames_KDEMask =
//            CountingFrame::allFromFolder(
//                config::framePath("mall"),
//                config::maskPath("mall"),
//                size,
//                mallCam.renderScaleMap(size)
//                ,roi
//                );
//
//    storeDataset("mall_kde", mallFrames_KDEMask, datasets);
////    storeDataset("mall_absdiff", mallFrames_AbsDiffMask, datasets);
//}
//
//
//void createPETS(DatasetCollection& datasets)
//{
//    RealisticCameraModel petsCamera("/work/sarandi/crowd/datasets/pets/calibration/View_001.xml");
//
//    Size size(320,240);
//    Mat1d scaleMap = petsCamera.renderScaleMap(size);
//
//    vector<string> subsetNames = {
//        "S1L1-1",
//        "S1L1-2",
//        "S1L2-1",
//        "S1L2-2",
//        "S2L1",
//        "S2L2",
//        "S2L3",
//        "S3MF1"};
//
//    vector<vector<CountingFrame>> petsFrameSets;
//
//    for (auto& subsetName : subsetNames)
//    {
//        auto frames
//                = CountingFrame::allFromFolder(parentFolder+"/pets/"+subsetName+"/", "masks_kde", size, scaleMap);
//
//        storeDataset("PETS " + subsetName, frames, datasets);
//        petsFrameSets.push_back(cvx::vectors::subVector(frames, 5));
//    }
//
//    storeDataset("PETS", cvx::vectors::flatten(petsFrameSets), datasets);
//}
//
//
//void createUCSD(DatasetCollection& datasets)
//{
//    auto parkFrames =
//            CountingFrame::allFromFolder(
//                parentFolder+"/park/",
//                "masks",
//                Size(320,240),
//                renderLinearScaleMap(0.2595, 1.0, 0.8291, 1.6666667, Size{320,240}));
//
//    storeDataset("park", parkFrames, datasets);
//
//    storeDataset("parkTraining 400-1200", cvx::vectors::subVector(parkFrames, 400, 1200), datasets);
//    storeDataset("parkTest 0-400, 1200-2000", cvx::vectors::concat(
//                     cvx::vectors::subVector(parkFrames, 0, 400),
//                     cvx::vectors::subVector(parkFrames, 1200, 2000)
//                    ), datasets);
//}
