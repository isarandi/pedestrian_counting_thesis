#ifndef CROWDCOUNTING_FLOWMOSAICKING_FLOWMOSAICCOUNTER_HPP_
#define CROWDCOUNTING_FLOWMOSAICKING_FLOWMOSAICCOUNTER_HPP_

#include <cvextra/ImageLoadingIterable.hpp>
#include <CrowdCounting/PersonLocations.hpp>
#include <CrowdCounting/LineCounting/FeatureSlices.hpp>
#include <CrowdCounting/OverallLineCounting/OverallLineCounter.hpp>
#include <MachineLearning/Regression.hpp>
#include <boost/optional.hpp>

namespace crowd { namespace linecounting {

class LineCountingSet;

class FlowMosaicDataset
{
public:
//    cvx::ImageLoadingIterable images;
//    cvx::ImageLoadingIterable cannyImages;
//    cvx::ImageLoadingIterable masks;

    std::vector<cv::Mat> images;
    std::vector<cv::Mat> cannyImages;
    std::vector<cv::Mat> masks;

    cv::Mat1d perspectiveWeights;
    cvx::BinaryMat roi;
    cvx::LineSegment seg;
    FeatureSlices slices;
    PersonLocations locations;

    static
    auto fromLineCountingSet(
            LineCountingSet const& lineSet
            ) -> FlowMosaicDataset;

};

class Blob{
public:
    cvx::BinaryMat blobMask;
    cv::Mat1d features;
    int desiredCrossings;
    double predictedCrossings;
    crowd::CrossingDir dir;
};

class FlowMosaicCounter : public OverallLineCounter
{
public:
    FlowMosaicCounter(
            Regression const& regression,
            int nLines,
            int minSegmentSize,
            double minMosaicSize,
            double segmentationThreshold,
            bool median = true)
        : leftRegression(regression.clone()),
          rightRegression(regression.clone()),
          nLines(nLines),
          minSegmentSize(minSegmentSize),
          minMosaicSize(minMosaicSize),
          segmentationThreshold(segmentationThreshold),
          median(median){}

    void train(std::vector<OverallLineCountingSet> const& trainingSet);
    auto predict(OverallLineCountingSet const& testSet) const -> PredictionWithConfidence2;

    auto getGroundTruth(OverallLineCountingSet const& testSet) const -> cv::Mat2d;
    auto getNumberOfLines() const -> int {return nLines;}

    CVX_CLONE_IN_DERIVED(FlowMosaicCounter)
    CVX_CONFIG_DERIVED(FlowMosaicCounter)

private:
    auto createBlobs(FlowMosaicDataset const& ds) const -> std::vector<Blob>;
    auto toLearningSet(
            std::vector<Blob> const& blobs
            ) const -> std::vector<LearningSet>;
    auto illustrate(
            FlowMosaicDataset const& dataset,
            std::vector<Blob> const& blobs,
            bool displayPredictionsToo
            ) const -> cv::Mat3b;

    stdx::cloned_unique_ptr<Regression> leftRegression;
    stdx::cloned_unique_ptr<Regression> rightRegression;
    int nLines;

    int minSegmentSize;
    double minMosaicSize;
    bool median;
    double segmentationThreshold;
};

} /* namespace linecounting */
} /* namespace crowd */

#endif /* CROWDCOUNTING_FLOWMOSAICKING_FLOWMOSAICCOUNTER_HPP_ */
