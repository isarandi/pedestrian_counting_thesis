#include <boost/iterator/iterator_facade.hpp>
#include <boost/range/irange.hpp>
#include <boost/range/iterator_range_core.hpp>
#include <cvextra/colors.hpp>
#include <cvextra/coords.hpp>
#include <cvextra/core.hpp>
#include <cvextra/cvret.hpp>
#include <cvextra/improc.hpp>
#include <cvextra/LoopRange.hpp>
#include <cvextra/mats.hpp>
#include <cvextra/strings.hpp>
#include <CrowdCounting/RegionCounting/CountingFrame.hpp>
#include <CrowdCounting/CrowdCountingUtils.hpp>
#include <CrowdCounting/RegionCounting/AllToAllRegionCounter.hpp>
#include <CrowdCounting/RegionCounting/Features/Extractors/Area.hpp>
#include <CrowdCounting/RegionCounting/Features/Extractors/EdgeMinkowski.hpp>
#include <CrowdCounting/RegionCounting/Features/Extractors/FilterBank.hpp>
#include <CrowdCounting/RegionCounting/Features/Extractors/GrayLevelCooccurrence.hpp>
#include <CrowdCounting/RegionCounting/Features/Extractors/LocalBinaryPatterns.hpp>
#include <CrowdCounting/RegionCounting/Features/Extractors/Perimeter.hpp>
#include <CrowdCounting/RegionCounting/Features/Extractors/PerimeterOriHistogram.hpp>
#include <CrowdCounting/RegionCounting/Features/Extractors/PerimeterToAreaRatio.hpp>
#include <CrowdCounting/RegionCounting/Features/Extractors/StatisticalLandscape.hpp>
#include <CrowdCounting/RegionCounting/Features/Impl/localBinaryPatternsImpl.hpp>
#include <CrowdCounting/RegionCounting/Features/MultiFeatureExtractor.hpp>
#include <CrowdCounting/RegionCounting/FrameCollection.hpp>
#include <MachineLearning/Ridge.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/core/mat.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <Run/setupDatasets.hpp>
#include <stdx/stdx.hpp>
#include <cmath>
#include <map>
#include <memory>
#include <string>
#include <vector>
#include <omp.h>

using namespace std;
using namespace cv;
using namespace cvx;
using namespace crowd;
using namespace boost::filesystem;

//Mat getTestMask()
//{
//    bpath parentFolder{"/work/sarandi/crowd/datasets/"};
//    vector<CountingFrame> mallFrames =
//            CountingFrame::allFromFolder(parentFolder/"mall", "masks_kde", Size(320,240), Mat());
//
//    Mat mask = mallFrames[20].getMask();
//
//    Mat s = cv::getStructuringElement(MORPH_ELLIPSE, Size(3,3), Point(1,1));
//    cv::dilate(mask, mask, s);
//    cv::erode(mask, mask, s);
//    cv::erode(mask, mask, s);
//    cv::dilate(mask, mask, s);
//
//    mask = cvxret::removeSmallConnectedComponents(mask, 50);
//    mask = cvxret::fillSmallHoles(mask, 20);
//    Mat grayFrame = cvret::cvtColor(mallFrames[0].getFrame(), COLOR_BGR2GRAY);
//
//    //Mat edges = cvret::Canny(grayFrame, 80, 150);
//
//    return mask;//Mat(mask, Rect(40,30, 40,30));
//}
//
//Mat getCellFeaturesOfGridRow(Mat featureMat, Size gridSize, int row)
//{
//    int nFeaturesPerCell = featureMat.cols / gridSize.area();
//    Rect featureMatRoi(row*gridSize.width*nFeaturesPerCell, 0, gridSize.width*nFeaturesPerCell, featureMat.rows);
//    return cvx::reshapeCols(featureMat(featureMatRoi).clone(), 0, nFeaturesPerCell);
//}
//
//Mat getCellGroundTruthOfGridRow(Mat outputMat, Size gridSize, int row)
//{
//    Rect outputMatRoi(row*gridSize.width, 0, gridSize.width, outputMat.rows);
//    return cvx::reshapeCols(outputMat(outputMatRoi).clone(), 0, 1);
//}
//
//void plotFeatureInGridRow(
//        Plotter& plotter,
//        Mat const& featureMat,
//        Mat const& outputMat,
//        string const& name,
//        Size gridSize,
//        int iFeature,
//        int row,
//        Scalar color)
//{
//    Mat X = getCellFeaturesOfGridRow(featureMat, gridSize, row);
//    Mat Y = getCellGroundTruthOfGridRow(outputMat, gridSize, row);
//
//    Mat1d noise(Y.size());
//    cv::randn(noise, 0, 0.08);
//    Mat noisyOutputs = Y + noise;
//
//    Mat1d column = X.col(iFeature).clone();
//    Mat1d noise2(column.size());
//
//    double min;
//    double max;
//    cv::minMaxLoc(column, &min, &max);
//
//    cv::randn(noise2, 0, (max-min)/15.0*0.2);
//    Mat noisyFeatures = column+noise2;
//
//    plotter.plot(noisyFeatures, noisyOutputs);//, name, color, Figure::GraphStyle::CROSS);
//}
//
//Mat drawCellScales(CountingFrame const& frame, Size gridSize)
//{
//    auto img = frame.getFrame().clone();
//    cvx::grid(img, gridSize, cvx::RED);
//
//    for (auto& cell : crowd::createGridRectangles(gridSize))
//    {
//        double scale = cvx::atRel(frame.getScaleMap(), cvx::center(cell));
//
//        Point absCenter = cvx::rel2abs(cvx::center(cell), img.size());
//        cvx::putTextCentered(img, cvx::str::format("%.2f", scale), absCenter, FONT_HERSHEY_PLAIN, 0.8, cvx::RED);
//    }
//
//    return img;
//
//}
//
//void plotComparisons(
//        FrameCollection const& coll1,
//        FrameCollection const& coll2,
//        Size gridSize,
//        int gridRowOfColl1,
//        MultiFeatureExtractor const& extr)
//{
//    auto alltoall = AllToAllCrowdCounter(gridSize, extr, Ridge(1e-3));
//    Mat featureMat1 = alltoall.getInputMatrix(coll1);
//    Mat outputMat1 = alltoall.buildOutputMatrix(coll1);
//
//    Mat featureMat2 = alltoall.getInputMatrix(coll2);
//    Mat outputMat2 = alltoall.buildOutputMatrix(coll2);
//
//    Mat1d scaleMap1 = coll1[0].getScaleMap();
//    Rectd gridCell = cvx::relativeGridRect(Point(gridSize.width/2, gridRowOfColl1), gridSize);
//    double scale1 = cvx::atRel(scaleMap1, cvx::center(gridCell));
//
//    Mat1d scaleMap2 = coll2[0].getScaleMap();
//
//    // look for matching row in other collection
//    auto rowRange = cvx::irange(gridSize.height);
//    int gridRowOfColl2 = *stdx::min_element_by(rowRange.begin(), rowRange.end(),
//    [&](int row){
//        Rectd gridCell = cvx::relativeGridRect(Point(gridSize.width/2,row), gridSize);
//        double scale2 = cvx::atRel(scaleMap2, cvx::center(gridCell));
//        return std::abs(scale1-scale2);
//    });
//
//    auto gridCell2 = cvx::relativeGridRect(Point(0,gridRowOfColl2), gridSize);
//    double scale2 = cvx::atRel(scaleMap2, cvx::center(gridCell2));
//
//    string name = cvx::str::format(
//                "%s row %d (%f) vs %s row %d (%f)",
//                coll1.getName(), gridRowOfColl1, scale1,
//                coll2.getName(), gridRowOfColl2, scale2);
//
//    for (int iFeature: cvx::irange(extr.getFeatureCount()))
//    {
//        Plotter p;
//        plotFeatureInGridRow(p, featureMat1, outputMat1, coll1.getName(), gridSize, iFeature, gridRowOfColl1, cvx::RED);
//        plotFeatureInGridRow(p, featureMat2, outputMat2, coll2.getName(), gridSize, iFeature, gridRowOfColl2, cvx::BLUE);
//
//        p.show();
//
////        path filepath = cvx::str::format("/work/sarandi/crowd/results/matlabPlots/m_%s.m", cvx::timestamp());
////        cvx::filesystem::write(filepath, f.matlabScript());
//
//        //cv::waitKey();
//    }
//
//}
//
//void testFeatures()
//{
//    //Persistence::off();
//    //omp_set_num_threads(1);
//
//    auto datasets = getDatasets();
//
//    FrameCollection& mall = *(datasets["mall_kde"]);
//    FrameCollection& ucsd = *(datasets["park"]);
//    FrameCollection& pets = *(datasets["PETS S1L1-1"]);
//
//    auto blobbased =
//                BlobBasedCounter(
//                MultiFeatureExtractor{ LocalBinaryPatterns(1,8,15,Size(1,1),true) },
//                NormalizedRegression(Ridge(1e-3)));
//
//    blobbased.showBlobs(mall);
//
//    crowd::localbinarypatterns::drawAll();
//
//    Size gridSize(8,8);
//
//    Mat illust1 = drawCellScales(mall[0], gridSize);
//    Mat illust2 = drawCellScales(pets[0], gridSize);
//
//    cv::imshow("mall", illust1);
//    cv::imshow("pets", illust2);
//
//    cv::waitKey();
//
//    auto extr = MultiFeatureExtractor{
//        Area(),
//        Perimeter(),
//        PerimeterOriHistogram(6),
//        PerimeterToAreaRatio(),
//        EdgeMinkowski(15),
//        StatisticalLandscape(31),
//        GrayLevelCooccurrence(),
//        FilterBank::LM(49),
//        LocalBinaryPatterns(1,8,15,Size(1,1),true),
//    };
//
//    plotComparisons(mall, pets, Size(8,8), 2, extr);
//
//
//}
