#include <BackgroundSegmentation/PixelFeatures/IntensityAndGradientFeatureExtractor.hpp>
#include <cvextra/cvret.hpp>
#include <cvextra/mats.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/core/mat.hpp>
#include <opencv2/core/types_c.h>
#include <opencv2/imgproc/imgproc.hpp>

using namespace std;
using namespace cv;
using namespace cvx;
using namespace crowd::bg;

Mat1d IntensityAndGradientFeatureExtractor::getFeatures(Mat const& img) const
{
    Mat1d features(img.total(), 3);

    Mat3b hsvImage = cvret::cvtColor(img, COLOR_BGR2HSV);
    Mat1b grayImg = cvret::cvtColor(img, COLOR_BGR2GRAY);

    Mat1s gradX = cvret::Sobel(grayImg, CV_16S, 1, 0);
    Mat1s gradY = cvret::Sobel(grayImg, CV_16S, 0, 1);

    int nPixels = hsvImage.total();
    for (int i = 0; i < nPixels; ++i)
    {
        Vec3b hsv = at(hsvImage,i);
        double v = hsv[2]/255.0f;

        cvx::mats::setRow(features, i, {v, at(gradX,i)/255.0f*gradientScale, at(gradY,i)/255.0f*gradientScale});
    }

    return features;
}
