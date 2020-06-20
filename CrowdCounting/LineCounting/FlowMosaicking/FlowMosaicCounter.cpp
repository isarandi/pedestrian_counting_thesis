#include <CrowdCounting/LineCounting/FlowMosaicking/FlowMosaicCounter.hpp>
#include <CrowdCounting/LineCounting/WindowBasedCounting.hpp>
#include <CrowdCounting/LineCounting/LineCountingSet.hpp>
#include <Flow/lineFlow.hpp>
#include <MachineLearning/LearningSet.hpp>

#include <Persistence.hpp>

#include <Python/EasyCallable.hpp>
#include <Python/EasyObject.hpp>
#include <Python/Pyplot.hpp>
#include <Python/PythonEnvironment.hpp>

#include <Run/config.hpp>
#include <Run/LineCounting/scenarios.hpp>

#include <cvextra/coords.hpp>
#include <cvextra/core.hpp>
#include <cvextra/cvret.hpp>
#include <cvextra/ComponentIterable.hpp>
#include <cvextra/improc.hpp>
#include <cvextra/io.hpp>
#include <cvextra/vectors.hpp>
#include <cvextra/LoopRange.hpp>
#include <cvextra/math.hpp>
#include <cvextra/strings.hpp>
#include <cvextra/visualize.hpp>
#include <cvextra/mats.hpp>
#include <cvextra/colors.hpp>
#include <CrowdCounting/LineCounting/LineCounter.hpp>
#include <opencv2/opencv.hpp>
#include <stdx/cloning.hpp>
#include <cassert>
#include <cmath>
#include <vector>

using namespace std;
using namespace cv;
using namespace cvx;
using namespace crowd;
using namespace crowd::linecounting;

auto getThickSlice(
        Mat image,
        LineSegment const& segment,
        cv::Range localRange,
        int width
        ) -> Mat
{
    int nLineSamples = segment.floorLength();
    Size roiSize{width, (int)localRange.size()};

    LineSegmentProperties segprop = segment.properties();
    Point2f p1 = segprop.localToGlobal(localRange.start);
    Point2f p2 = segprop.localToGlobal(localRange.end);

    //--- Rotated rect around line in default coord.sys.
    RotatedRect roiRect(
            cvx::point2f(p1+p2)/2,
            roiSize,
            cvx::toDegrees(segment.angleFromVerticalRadians()));
    Mat extractorMatrix = crowd::lineopticalflow::details::getRotatedRectExtractorMatrix(roiRect);
    return cvret::warpAffine(image, extractorMatrix, roiSize);
}

auto segmentBlobBasedOnCumulativeSpeed(
        Mat1d const& flowX,
        BinaryMat const& mask,
        OutputArray labelsOut,
        double threshold) -> int
{
    labelsOut.create(mask.size(), CV_32S);
    Mat1i blobLabels = labelsOut.getMat();
    blobLabels.setTo(0);

    Mat1d maskedFlowX = flowX.clone();
    maskedFlowX.setTo(0, mask==0);

    int iBlob = 1;
    double cumulSpeed = 0.0;

    for (int iFrame : cvx::irange(mask.rows))
    {
        if (cv::countNonZero(mask.row(iFrame)) == 0)
        {
            continue;
        }

        double sumSpeed = std::abs(cv::sum(maskedFlowX.row(iFrame))[0]);
        if (std::floor(cumulSpeed/threshold) < std::floor((cumulSpeed+sumSpeed)/threshold)
            && cumulSpeed != 0)
        {
            ++iBlob;
        }

        blobLabels.row(iFrame).setTo(iBlob, mask.row(iFrame));
        cumulSpeed += sumSpeed;
    }
    int nLabels = iBlob+1;

    return nLabels;
}


Mat1d customMultiply(Mat1d m1, Mat1d m2)
{
    assert(m1.size() == m2.size());

    Mat1d result{m1.size()};

    for (Point const& p : cvx::points(result))
    {
        result(p) = m1(p)*m2(p);
    }

    return result;
}

auto crowd::linecounting::FlowMosaicCounter::
createBlobs(FlowMosaicDataset const& dataset) const -> std::vector<Blob>
{
    cout << "Creating blobs..." << endl;

    FeatureSlices slices = dataset.slices;
    Mat1f flowX = cvret::extractChannel(slices.flow, 0);
    Mat1f flowY = cvret::extractChannel(slices.flow, 1);

    if (median)
    {
        flowX = cvxret::medianFilter(flowX, Size{3,3});
    }
    cv::GaussianBlur(flowX, flowX, Size{5,5}, 0.02, 1.1);
    slices.flow = cvret::merge({flowX,flowY});
    Mat1b directions =
            crowd::linecounting::LineLearningRepresenter::
            segmentMovementStatic(slices.flow, slices.image, segmentationThreshold);
    directions.setTo(0, slices.stencil==0);

    vector<Blob> blobs;

    for (auto dir : std::vector<CrossingDir>{CrossingDir::LEFTWARD, CrossingDir::RIGHTWARD})
    {
        BinaryMat singleDirectionMask = (directions == (int)dir);

        for (BinaryMat rawComponentMask : cvx::connectedComponentMasks(singleDirectionMask, 8))
        {

            if (cv::countNonZero(rawComponentMask) < minSegmentSize)
            {
                cout << "outside " << cv::countNonZero(rawComponentMask) << endl;
                continue;
            }

            Mat1i cumulBlobLabels;
            int nCumulBlobLabels =
                    segmentBlobBasedOnCumulativeSpeed(
                            flowX,
                            rawComponentMask,
                            cumulBlobLabels,
                            minMosaicSize);

            for (int iCumulBlobLabel : cvx::irange(1,nCumulBlobLabels))
            {
                for (BinaryMat blobMask : cvx::connectedComponentMasks(cumulBlobLabels == iCumulBlobLabel, 8))
                {
                    if (cv::countNonZero(blobMask) < minSegmentSize)
                    {
                        cout << "inside " << cv::countNonZero(blobMask) << endl;
                        continue;
                    }

                    double A = 0;
                    double B = 0;

                    for (int iFrame : cvx::irange(blobMask.rows))
                    {
                        Mat upscaledMask =
                                cvret::resize(
                                        dataset.masks[iFrame],
                                        dataset.cannyImages[iFrame].size(),
                                        0,0, INTER_NEAREST);

                        for (BinaryMat thinBlob : cvx::connectedComponentMasks(blobMask.row(iFrame), 8))
                        {
                            auto box = cvx::boundingBoxOfBinary(thinBlob);
                            Range segmentRange = Range{box.x, box.x+box.width};

                            double meanSpeed = std::abs(cv::mean(flowX.row(iFrame), thinBlob)[0]);
                            int width = (int)meanSpeed;
                            Mat1d cannyExtracted =
                                    cvret::convertType(
                                            getThickSlice(
                                                    dataset.cannyImages[iFrame],
                                                    dataset.seg,
                                                    segmentRange,
                                                    width), CV_64F)/255.;

                            Mat1d maskExtracted =
                                    cvret::convertType(
                                            getThickSlice(
                                                    upscaledMask,
                                                    dataset.seg,
                                                    segmentRange,
                                                    width), CV_64F)/255.;

                            Mat1d weightsExtracted =
                                    getThickSlice(dataset.perspectiveWeights, dataset.seg, segmentRange, width);

                            A += cv::sum(customMultiply(customMultiply(weightsExtracted,weightsExtracted), maskExtracted))[0];
                            B += cv::sum(customMultiply(customMultiply(weightsExtracted,maskExtracted), cannyExtracted))[0];
                        }
                    }

                    int crossingsInBlob = 0;
                    for (auto const& crossing : dataset.locations.getLineCrossings(dataset.seg))
                    {
                        int localPos = (int)crossing.localPos;
                        if (localPos >=0 &&
                                localPos < blobMask.cols &&
                                crossing.frameId >= 0 &&
                                crossing.frameId < blobMask.rows &&
                                blobMask(crossing.frameId, localPos)==255 &&
                                crossing.dir == dir)
                        {
                            ++crossingsInBlob;
                        }
                    }

                    cout << "in blob: " << crossingsInBlob << endl;

                    if (crossingsInBlob > 10)
                    {
                        cv::imshow(" ", blobMask.t());
                        cv::waitKey();
                    }

                    blobs.push_back(
                            Blob{
                                blobMask,
                                cvx::m({{A,B}}),
                                crossingsInBlob,
                                0.0,
                                dir});
                }
            }
        }
    }
    return blobs;
}

auto FlowMosaicCounter::
toLearningSet(std::vector<Blob> const& blobs) const -> std::vector<LearningSet>
{
    LearningSet leftSet;
    LearningSet rightSet;

    for (auto& blob : blobs)
    {
        if (blob.dir == CrossingDir::LEFTWARD)
        {
            leftSet.verticalAdd({
                blob.features,
                cvx::m({{(double)blob.desiredCrossings}})
            });
        } else {
            rightSet.verticalAdd({
                blob.features,
                cvx::m({{(double)blob.desiredCrossings}})
            });
        }
    }

    return {leftSet, rightSet};
}

auto FlowMosaicCounter::
illustrate(
        FlowMosaicDataset const& dataset,
        std::vector<Blob> const& blobs,
        bool displayPredictionsToo
        ) const -> Mat3b
{
    Mat3b illust = cvx::visu::darkened(dataset.slices.image.t());

    for (auto const& blob : blobs)
    {
        cvx::visu::highlightOnto(dataset.slices.image.t(), blob.blobMask.t(), illust, blob.dir == CrossingDir::LEFTWARD ? cvx::GREEN : cvx::RED);
        Rect box = cvx::boundingBoxOfBinary(blob.blobMask);
        Point2d p = cvx::point2d(cvx::center(box));

        cvx::putTextCairo(
                illust,
                std::to_string(blob.desiredCrossings),
                {p.y, p.x},
                "arial",
                20, cvx::GREEN);

        if (displayPredictionsToo)
        {
            cvx::putTextCairo(
                    illust,
                    stdx::to_string(blob.predictedCrossings, 2),
                    {p.y, p.x+20},
                    "arial",
                    20, cvx::RED);
        }
    }

    return illust;
}

void crowd::linecounting::FlowMosaicCounter::
train(std::vector<OverallLineCountingSet> const& trainingSet)
{
    auto lineSets = LineCountingSet::parallelLineSets(trainingSet, nLines);

    string name =
            cvx::str::format(
                    "Flow Mosaicking, %d %f %d %d %f",
                    minSegmentSize,
                    minMosaicSize,
                    median ? 1:0,
                    nLines,
                    segmentationThreshold) +
            cvx::configfile::json_str(cvx::configfile::describeCollection(trainingSet));

    auto mats = Persistence::loadOrComputeMats(name, {"XL", "YL", "XR", "YR"}, [&]()->std::vector<cv::Mat>
    {

        LearningSet leftSet;
        LearningSet rightSet;

        for (auto const& lineSet : lineSets)
        {
            FlowMosaicDataset ds = FlowMosaicDataset::fromLineCountingSet(lineSet);
            auto blobs = createBlobs(ds);

//            cv::imshow("i", illustrate(ds, blobs, false));
//            cv::waitKey();

            auto learningSets = toLearningSet(blobs);
            leftSet.verticalAdd(learningSets[0]);
            rightSet.verticalAdd(learningSets[1]);
        }

        return std::vector<Mat>{leftSet.X, leftSet.Y, rightSet.X, rightSet.Y};
    });

    LearningSet leftSet{mats[0], mats[1]};
    LearningSet rightSet{mats[2], mats[3]};

    pyx::Pyplot plt;
    plt.plot(leftSet.X.col(0), leftSet.Y, "g+");
    plt.plot(leftSet.X.col(1), leftSet.Y, "ro");
    plt.showAndClose();

    pyx::Pyplot plt2;
    plt2.plot(rightSet.X.col(0), rightSet.Y, "g+");
    plt2.plot(rightSet.X.col(1), rightSet.Y, "ro");
    plt2.showAndClose();

    leftRegression->train(leftSet);
    rightRegression->train(rightSet);
}

auto FlowMosaicCounter::
getGroundTruth(OverallLineCountingSet const& testSet) const -> Mat2d
{
    Size frameSize = config::frames(testSet.datasetName)[0].size();

    Mat1d result;
    for (auto const& line : crowd::getStandardLines(frameSize, nLines))
    {
        LineCountingSet testSequence{testSet.datasetName, line, testSet.frameRange};
        Mat1d desiredPerFrame = testSequence.loadLocations().getInstantFlow(line);
        cvx::hconcat(result, desiredPerFrame, result);
    }

    return result.reshape(2);
}

auto FlowMosaicCounter::
predict(OverallLineCountingSet const& testSet) const -> PredictionWithConfidence2
{
    auto lineSets = LineCountingSet::parallelLineSets({testSet}, nLines);

    int nFrames =
            (testSet.frameRange == cv::Range::all()) ?
                    config::frames(testSet.datasetName).size() :
                    testSet.frameRange.size();

    Mat1d leftResults{nFrames, nLines, 0.0};
    Mat1d rightResults{nFrames, nLines, 0.0};

    int iLine = 0;
    for (auto const& lineSet : lineSets)
    {
        FlowMosaicDataset ds = FlowMosaicDataset::fromLineCountingSet(lineSet);
        auto blobs = createBlobs(ds);

        for (auto& blob : blobs)
        {
            Rect blobBox = cvx::boundingBoxOfBinary(blob.blobMask);

            Range blobRange{blobBox.y, blobBox.y+blobBox.height};

            if (blob.dir == CrossingDir::LEFTWARD)
            {
                blob.predictedCrossings = leftRegression->predict(blob.features)(0,0);
                leftResults.rowRange(blobRange).col(iLine) += blob.predictedCrossings/blobBox.height;
            } else {
                blob.predictedCrossings = rightRegression->predict(blob.features)(0,0);
                rightResults.rowRange(blobRange).col(iLine) += blob.predictedCrossings/blobBox.height;
            }
        }
//        cv::imshow("i", illustrate(ds, blobs, true));
//        cv::waitKey();
        ++iLine;
    }

    Mat2d mean = cvret::merge({leftResults, rightResults});
    return {mean, Mat2d::ones(mean.size())};
}

auto FlowMosaicDataset::
fromLineCountingSet(LineCountingSet const& lineSet) -> FlowMosaicDataset
{
    return FlowMosaicDataset {
        cvx::vectors::subVector(config::frames_immediate(lineSet.datasetName), lineSet.frameRange),
        cvx::vectors::subVector(config::cannys_immediate(lineSet.datasetName), lineSet.frameRange),
        cvx::vectors::subVector(config::masks_immediate(lineSet.datasetName), lineSet.frameRange),
        cvret::divide(1.0, config::scaleMap(lineSet.datasetName)),
        config::roiStencil(lineSet.datasetName),
        lineSet.segment,
        lineSet.loadSlices(),
        config::locations(lineSet.datasetName).applyStencil(config::roiStencil(lineSet.datasetName)).betweenFrames(lineSet.frameRange)};

//    config::frames(lineSet.datasetName).range(lineSet.frameRange),
//    config::cannys(lineSet.datasetName).range(lineSet.frameRange),
//    config::masks(lineSet.datasetName).range(lineSet.frameRange),
//    cvret::divide(1.0, config::scaleMap(lineSet.datasetName)),
//    config::roiStencil(lineSet.datasetName),
//    lineSet.segment,
//    lineSet.loadSlices(),
//    config::locations(lineSet.datasetName).applyStencil(config::roiStencil(lineSet.datasetName)).betweenFrames(lineSet.frameRange)};
}


auto FlowMosaicCounter::
describe() const -> boost::property_tree::ptree
{
    boost::property_tree::ptree pt;
    pt.put("type", "FlowMosaicCounter");
    pt.put("nLines", nLines);
    pt.put("minSegmentSize", minSegmentSize);
    pt.put("minMosaicSize", minMosaicSize);
    pt.put("segmentationThreshold", segmentationThreshold);
    pt.put_child("regression", leftRegression->describe());
    return pt;
}

auto FlowMosaicCounter::
create(boost::property_tree::ptree const& pt) -> std::unique_ptr<FlowMosaicCounter>
{
    return stdx::make_unique<FlowMosaicCounter>(
            *Regression::create(pt.get_child("regression")),
            pt.get<int>("nLines"),
            pt.get<int>("minSegmentSize"),
            pt.get<double>("minMosaicSize"),
            pt.get<double>("segmentationThreshold")

    );
}
