#include <cvextra/core.hpp>
#include <cvextra/improc.hpp>
#include <cvextra/mats.hpp>
#include <cvextra/LoopRange.hpp>
#include <cvextra/strings.hpp>
#include <CrowdCounting/RegionCounting/Features/Extractors/PerimeterOriHistogram.hpp>
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

PerimeterOriHistogram::
PerimeterOriHistogram(int _nBins)
    : nBins(_nBins)
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

    for (int iRotation : cvx::irange(nBins))
    {
        Mat rotatedKernel =
        		cvxret::rotate(
        				horizontalGaussKernel,
						centerPoint,
						iRotation*180.0/nBins,
						AngleUnit::DEGREES);

        filters.push_back(rotatedKernel);
    }
}

auto PerimeterOriHistogram::
extract(
		PreprocessedFrame const& frame,
		Rectd relativeRect
		) const -> std::vector<double>
{
    BinaryMat edgesPart = cvx::extractRelativeRoi(frame.mask, relativeRect);
    BinaryMat perimeter = cvxret::outline(edgesPart);

    Mat scalePart = cvx::extractRelativeRoi(frame.scaleMap, relativeRect);
    Mat weights = 1.0/scalePart;
    weights.setTo(0, 255-perimeter);

    Mat1d perimeterAsDouble = perimeter;

    return crowd::weightedMaxResponseHistogram(perimeterAsDouble, weights, filters);
}

auto PerimeterOriHistogram::
getFeatureCount() const -> int
{
    return nBins;
}

auto PerimeterOriHistogram::
getNames() const -> std::vector<string>
{
    return crowd::createIndexSuffixedVersions("perimeter orientation histogram ", nBins);
}

auto PerimeterOriHistogram::
getDescription() const -> string
{
    stringstream ss;

    ss << "Bin count: " << nBins << endl;
    ss << "Gauss kernel size: " << kernelSideLength << "x" << kernelSideLength << endl;
    ss << "Gauss kernel sigma1: " << horizontalSigma << endl;
    ss << "Gauss kernel sigma2: " << verticalSigma;

    return "Perimeter orientation histogram\n" + cvx::str::indentBlock(ss.str());
}

auto PerimeterOriHistogram::
describe() const -> boost::property_tree::ptree
{
	boost::property_tree::ptree pt;
	pt.put("type", "PerimeterOriHistogram");
	pt.put("nBins", nBins);
	return pt;
}

auto PerimeterOriHistogram::
create(boost::property_tree::ptree const& pt) -> std::unique_ptr<PerimeterOriHistogram>
{
	return stdx::make_unique<PerimeterOriHistogram>(pt.get<int>("nBins"));
}
