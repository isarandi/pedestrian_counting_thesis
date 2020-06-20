#include <cvextra/core.hpp>
#include <cvextra/cvret.hpp>
#include <cvextra/LoopRange.hpp>
#include <cvextra/strings.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/core/mat.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/core/types_c.h>
#include <opencv2/imgproc/imgproc.hpp>
#include <stdx/stdx.hpp>
#include "features.hpp"
#include <cmath>
#include <iterator>
#include <string>
#include <vector>

using namespace std;
using namespace cv;
using namespace cvx;
using namespace crowd;

auto crowd::
createIndexSuffixedVersions(string const& name, int count) -> vector<string>
{
    int nDigits = std::floor(std::log10(count))+1;

    vector<string> result;
    for (int i = 0; i < count; ++i)
    {
        result.push_back(name + cvx::str::zeropad(i, nDigits));
    }
    return result;
}

double crowd::
weightedPixelCount(InputArray _src, InputArray _weights)
{
    BinaryMat binImg = _src.getMat();
    Mat1d weights = _weights.getMat();

    double weightedSum = 0;
    for (Point p : cvx::points(binImg))
    {
        if (binImg(p) != 0)
        {
            weightedSum += weights(p);
        }
    }

    return weightedSum;
}

auto crowd::
weightedMaxResponseHistogram(
        InputArray input,
        InputArray _weights,
        std::vector<cv::Mat> const& filters
		) -> vector<double>
{
    vector<double> histogram(filters.size(), 0);

    if (cv::countNonZero(_weights) == 0)
    {
        return histogram;
    }

    // filter image with all filters
    vector<Mat1d> filteredImages;
    for (auto& filter : filters)
    {
        Mat1d filteredImage = cvret::filter2D(input, CV_64F, filter, Point(-1,-1), 0, BORDER_CONSTANT);
        filteredImages.push_back(filteredImage);
    }

    // build histogram
    Mat1d weights = _weights.getMat();

    for (Point p : cvx::points(input.size()))
    {
        if (weights(p) != 0)
        {
            auto maxIter = stdx::max_element_by(
            		filteredImages.begin(),
					filteredImages.end(),
					[&](Mat1d const& filteredImage) {
						return filteredImage(p);
					});

            int maxIndex = maxIter - filteredImages.begin();
            histogram[maxIndex] += weights(p);
        }
    }

    return histogram;
}
