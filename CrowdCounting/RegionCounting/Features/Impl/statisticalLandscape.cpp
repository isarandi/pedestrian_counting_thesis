#include "features.hpp"
#include <boost/range/irange.hpp>
#include <cvextra/core.hpp>
#include <cvextra/improc.hpp>
#include <cvextra/LoopRange.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/core/mat.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/core/types_c.h>
#include <vector>

using namespace std;
using namespace cv;
using namespace cvx;

static void
meanOfComponentMeans(
        InputArray _img,
        InputArray thresholded,
        int& nComponentsOut,
        double& meanOfComponentMeansOut)
{
    Mat1i labelImage;
    int nComponents =
    		cvx::connectedComponents(thresholded, labelImage, 8, CV_32S) - 1;

    if (nComponents == 0)
    {
        nComponentsOut = 0;
        meanOfComponentMeansOut = 0;
        return;
    }

    Mat1d img = _img.getMat();

    vector<double> componentSums(nComponents);
    vector<int> componentNPixels(nComponents);

    for (Point p : cvx::points(labelImage))
    {
        int label = labelImage(p);

        if (label > 0)
        {
            componentSums[label-1] += img(p);
            ++componentNPixels[label-1];
        }
    }

    double sumOfComponentMeans = 0;
    for (int iComponent : cvx::irange(nComponents))
    {
        sumOfComponentMeans +=
        		componentSums[iComponent]/componentNPixels[iComponent];
    }

    nComponentsOut = nComponents;
    meanOfComponentMeansOut = sumOfComponentMeans / nComponents;
}

auto crowd::
statisticalLandscape(
		InputArray src,
		vector<double> const& thresholds
		) -> vector<double>
{
    vector<double> result;
    result.reserve(thresholds.size()*4);

    BinaryMat thresholded;
    for (double threshold : thresholds)
    {
        cv::compare(src, Scalar(threshold), thresholded, CMP_GE);

        int nUpperComponents;
        double meanOfUpperComponentMeans;
        meanOfComponentMeans(src, thresholded, nUpperComponents, meanOfUpperComponentMeans);

        int nLowerComponents;
        double meanOfLowerComponentMeans;
        meanOfComponentMeans(src, 255-thresholded, nLowerComponents, meanOfLowerComponentMeans);

        result.push_back(nUpperComponents);
        result.push_back(meanOfUpperComponentMeans-threshold);
        result.push_back(nLowerComponents);
        result.push_back(threshold-meanOfLowerComponentMeans);
    }

    return result;
}
