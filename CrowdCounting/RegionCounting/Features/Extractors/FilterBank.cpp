#include <cvextra/core.hpp>
#include <cvextra/cvret.hpp>
#include <cvextra/improc.hpp>
#include <cvextra/mats.hpp>
#include <cvextra/vectors.hpp>
#include <CrowdCounting/RegionCounting/Features/Extractors/FilterBank.hpp>
#include <CrowdCounting/RegionCounting/Features/Impl/features.hpp>
#include <CrowdCounting/RegionCounting/Features/PreprocessedFrame.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/core/mat.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/core/types_c.h>
#include <opencv2/imgproc/imgproc.hpp>
#include <algorithm>
#include <cmath>
#include <string>
#include <vector>

using namespace std;
using namespace cv;
using namespace cvx;
using namespace crowd;

FilterBank::
FilterBank(
		vector<Mat> _filters,
		vector<string> _filterNames,
		bool _masked)
    : filters(_filters)
    , masked(_masked)
{
    if (filterNames.empty())
    {
        filterNames = crowd::createIndexSuffixedVersions("Filter Bank", filters.size());
    } else {
        filterNames = _filterNames;
    }
}

auto FilterBank::
extract(
		PreprocessedFrame const& frame,
		Rectd relativeRect
		) const -> vector<double>
{
    Mat1b maskPart =
    		masked ?
    				cvx::extractRelativeRoi(frame.mask, relativeRect) :
					Mat1b{maskPart.size(), 255};

    Mat1d grayPart = cvx::extractRelativeRoi(frame.grayFrame, relativeRect);
    return getAverageResponses(grayPart, maskPart);
}

auto FilterBank::
getFeatureCount() const -> int
{
    return filters.size();
}

auto FilterBank::
getNames() const -> vector<string>
{
    return filterNames;
}

auto FilterBank::
getDescription() const -> string
{
    return "Filter Bank";
}

auto getCenterPart(Mat src, Size middleSize) -> Mat
{
    Point offset = (src.size()-middleSize)/2;
    return src(Rect(offset, middleSize));
}

auto FilterBank::
createLMFilters(int s) -> vector<Mat>
{
    Size filterSize{s,s};

    int enlargedS = std::ceil(s * std::sqrt(2));
    Size enlargedFilterSize{enlargedS,enlargedS};

    vector<double> basicSigmas{1, std::sqrt(2), 2, 2 * std::sqrt(2)};
    vector<double> edgeBarSigmaXs(basicSigmas.begin(), basicSigmas.begin()+3);

    int nOrientations = 6;
    vector<Mat> edgeFilters;
    vector<Mat> barFilters;

    Point2d enlargedFilterCenter{enlargedS/2.0, enlargedS/2.0};

    // Create edge and bar filters
    for (double sigmaX : edgeBarSigmaXs)
    {
        double sigmaY = 3*sigmaX;

        Mat verticalGauss = cvx::getGaussianKernel(enlargedFilterSize, sigmaX, sigmaY);
        Mat verticalEdge = cvret::Sobel(verticalGauss, CV_64F, 1, 0, 3, 1, 0, BORDER_CONSTANT);
        Mat verticalBar = cvret::Sobel(verticalGauss, CV_64F, 2, 0, 3, 1, 0, BORDER_CONSTANT);

        for (int iAngle : cvx::irange(nOrientations))
        {
            double angleInDegrees = (180.0 * iAngle)/nOrientations;

            edgeFilters.push_back(
                        getCenterPart(
                            cvxret::rotate(
                                verticalEdge,
                                enlargedFilterCenter,
                                angleInDegrees),
                            filterSize));

            barFilters.push_back(
                        getCenterPart(
                            cvxret::rotate(
                                verticalBar,
                                enlargedFilterCenter,
                                angleInDegrees),
                            filterSize));
        }
    }

    // Create Laplacians
    vector<double> laplaceSigmas =
            cvx::vectors::concat(
                basicSigmas,
                cvx::vectors::transform<double>(basicSigmas, [](double d){return 3*d;}));
    std::sort(laplaceSigmas.begin(), laplaceSigmas.end());

    vector<Mat> laplacianOfGaussianFilters;
    for (double sigma : laplaceSigmas)
    {
        Mat gaussian = cvx::getGaussianKernel(Size{s+2,s+2}, sigma, sigma);
        Mat laplacianOfGaussian = cvret::Laplacian(gaussian, CV_64F, 3, 1, 0, BORDER_CONSTANT);
        laplacianOfGaussianFilters.push_back(getCenterPart(laplacianOfGaussian, filterSize));
    }

    // Create Gaussians
    vector<Mat> gaussianFilters;
    vector<double> gaussianSigmas = basicSigmas;

    for (double sigma : gaussianSigmas)
    {
        Mat gaussian = cvx::getGaussianKernel(filterSize, sigma, sigma);
        gaussianFilters.push_back(gaussian);
    }

    vector<Mat> filters = cvx::vectors::flatten<Mat>
            ({edgeFilters, barFilters, laplacianOfGaussianFilters, gaussianFilters});

    return filters;
}

auto FilterBank::
createCircLMFilters(int s) -> vector<Mat>
{
    Size filterSize{s,s};

    int enlargedS = std::ceil(s * std::sqrt(2));
    Size enlargedFilterSize{enlargedS,enlargedS};

    vector<double> basicSigmas{1, std::sqrt(2), 2, 2 * std::sqrt(2)};
    vector<double> edgeBarSigmaXs(basicSigmas.begin(), basicSigmas.begin()+3);

    int nOrientations = 6;
    vector<Mat> edgeFilters;
    vector<Mat> barFilters;

    Point2d enlargedFilterCenter{enlargedS/2.0, enlargedS/2.0};

    // Create edge and bar filters
    for (double sigmaX : edgeBarSigmaXs)
    {
        double sigmaY = 3*sigmaX;

        Mat verticalGauss = cvx::getGaussianKernel(enlargedFilterSize, sigmaX, sigmaY);
        Mat verticalEdge = cvret::Sobel(verticalGauss, CV_64F, 1, 0, 3, 1, 0, BORDER_CONSTANT);
        Mat verticalBar = cvret::Sobel(verticalGauss, CV_64F, 2, 0, 3, 1, 0, BORDER_CONSTANT);

        for (int iAngle : cvx::irange(nOrientations))
        {
            double angleInDegrees = (180.0 * iAngle)/nOrientations;

            edgeFilters.push_back(
                        getCenterPart(
                            cvxret::rotate(
                                verticalEdge,
                                enlargedFilterCenter,
                                angleInDegrees),
                            filterSize));

            barFilters.push_back(
                        getCenterPart(
                            cvxret::rotate(
                                verticalBar,
                                enlargedFilterCenter,
                                angleInDegrees),
                            filterSize));
        }
    }

    // Create Laplacians
    vector<double> laplaceSigmas =
            cvx::vectors::concat(
                basicSigmas,
                cvx::vectors::transform<double>(basicSigmas, [](double d){return 3*d;}));
    std::sort(laplaceSigmas.begin(), laplaceSigmas.end());

    vector<Mat> laplacianOfGaussianFilters;
    for (double sigma : laplaceSigmas)
    {
        Mat gaussian = cvx::getGaussianKernel(Size{s+2,s+2}, sigma, sigma);
        Mat laplacianOfGaussian = cvret::Laplacian(gaussian, CV_64F, 3, 1, 0, BORDER_CONSTANT);
        laplacianOfGaussianFilters.push_back(getCenterPart(laplacianOfGaussian, filterSize));
    }

    // Create Gaussians
    vector<Mat> gaussianFilters;
    vector<double> gaussianSigmas = basicSigmas;

    for (double sigma : gaussianSigmas)
    {
        Mat gaussian = cvx::getGaussianKernel(filterSize, sigma, sigma);
        gaussianFilters.push_back(gaussian);
    }

    vector<Mat> filters = cvx::vectors::flatten<Mat>
            ({laplacianOfGaussianFilters, gaussianFilters});

    return filters;
}

auto FilterBank::
getResponses(InputArray input) const -> vector<Mat>
{
    vector<Mat> responses;

    for (auto& filter : filters)
    {
        responses.push_back(cvret::filter2D(input, CV_64F, filter));
    }
    return responses;
}

void FilterBank::
getResponses(InputArray input, vector<Mat>& responses) const
{
    int i=0;
    for (auto& filter : filters)
    {
        cv::filter2D(input, responses[i], CV_32F, filter);
        ++i;
    }
}

auto FilterBank::
getAverageResponses(InputArray input, InputArray mask) const -> vector<double>
{
    vector<double> averageResponses;
    for (auto& responseImage : getResponses(input))
    {
        averageResponses.push_back(cv::mean(responseImage, mask)[0]);
    }
    return averageResponses;
}

auto FilterBank::
LM(int oddSize) -> FilterBank
{
    vector<string> names;
    cvx::vectors::push_back_all(names, crowd::createIndexSuffixedVersions("LM edge filter", 6));
    cvx::vectors::push_back_all(names, crowd::createIndexSuffixedVersions("LM bar filter", 6));
    cvx::vectors::push_back_all(names, crowd::createIndexSuffixedVersions("LM Lapl-of-Gauss filter", 8));
    cvx::vectors::push_back_all(names, crowd::createIndexSuffixedVersions("LM Gauss filter", 4));

    return FilterBank{FilterBank::createLMFilters(oddSize), names};
}

auto FilterBank::
CircLM(int oddSize) -> FilterBank
{
    vector<string> names;
    cvx::vectors::push_back_all(names, crowd::createIndexSuffixedVersions("LM Lapl-of-Gauss filter", 8));
    cvx::vectors::push_back_all(names, crowd::createIndexSuffixedVersions("LM Gauss filter", 4));

    return FilterBank{FilterBank::createCircLMFilters(oddSize), names};
}

auto FilterBank::
describe() const -> boost::property_tree::ptree
{
	boost::property_tree::ptree pt;
	pt.put("type", "FilterBank");
	return pt;
}

auto FilterBank::
create(boost::property_tree::ptree const& pt) -> std::unique_ptr<FilterBank>
{
	return stdx::make_unique<FilterBank>(LM(49));
}
