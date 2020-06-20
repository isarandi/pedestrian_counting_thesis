#include <cvextra/core.hpp>
#include <cvextra/improc.hpp>
#include <cvextra/mats.hpp>
#include <cvextra/strings.hpp>
#include <CrowdCounting/RegionCounting/Features/Extractors/EdgeOriHistogram.hpp>
#include <CrowdCounting/RegionCounting/Features/Impl/features.hpp>
#include <CrowdCounting/RegionCounting/Features/PreprocessedFrame.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/core/mat.hpp>
#include <sstream>
#include <string>
#include <vector>

using namespace std;
using namespace cv;
using namespace cvx;
using namespace crowd;

EdgeOriHistogram::EdgeOriHistogram(int _nBins, bool masked)
    : nBins(_nBins)
    , masked(masked)
{
    // create oriented Gaussian filters
    kernelSideLength = 15;
    horizontalSigma = kernelSideLength*0.3;
    verticalSigma = kernelSideLength*0.04;

    Mat horizontalGaussKernel =
            cvx::getGaussianKernel(
                {kernelSideLength,kernelSideLength},
                horizontalSigma,
                verticalSigma);

    Point2d centerPoint{kernelSideLength/2.0, kernelSideLength/2.0};

    for (int iRotation = 0; iRotation < nBins; ++iRotation)
    {
        Mat rotatedKernel = cvxret::rotate(horizontalGaussKernel, centerPoint, iRotation*180.0/nBins);
        filters.push_back(rotatedKernel);
    }
}

std::vector<double> EdgeOriHistogram::extract(PreprocessedFrame const& frame, Rectd relativeRect) const
{
    BinaryMat edgesPart = cvx::extractRelativeRoi(frame.edges, relativeRect);

    if (masked)
    {
        BinaryMat maskPart = cvx::extractRelativeRoi(frame.mask, relativeRect);
        edgesPart = cv::min(edgesPart, maskPart);
    }

    Mat scalePart = cvx::extractRelativeRoi(frame.scaleMap, relativeRect);
    Mat weights = 1.0/scalePart;
    weights.setTo(0, 255-edgesPart);

    Mat1d doubleEdgesPart = edgesPart;

    return crowd::weightedMaxResponseHistogram(doubleEdgesPart, weights, filters);
}

int EdgeOriHistogram::getFeatureCount() const
{
    return nBins;
}

std::vector<string> EdgeOriHistogram::getNames() const
{
    return crowd::createIndexSuffixedVersions("edge orientation histogram ", nBins);
}

string EdgeOriHistogram::getDescription() const
{
    stringstream ss;

    ss << "Bin count: " << nBins << endl;
    ss << "Masked: " << masked << endl;
    ss << "Gauss kernel size: " << kernelSideLength << "x" << kernelSideLength << endl;
    ss << "Gauss kernel sigma1: " << horizontalSigma << endl;
    ss << "Gauss kernel sigma2: " << verticalSigma;

    return "Edge orientation histogram\n" + cvx::str::indentBlock(ss.str());
}

auto EdgeOriHistogram::
describe() const -> boost::property_tree::ptree
{
	boost::property_tree::ptree pt;
	pt.put("type", "EdgeOriHistogram");
	pt.put("nBins", nBins);
	pt.put("masked", masked);
	return pt;
}

auto EdgeOriHistogram::
create(boost::property_tree::ptree const& pt) -> std::unique_ptr<EdgeOriHistogram>
{
	return stdx::make_unique<EdgeOriHistogram>(pt.get<int>("nBins"), pt.get<bool>("masked"));
}
