#include <BackgroundSegmentation/PixelFeatures/IntensityFeatureExtractor.hpp>
#include <cvextra/cvret.hpp>
#include <cvextra/mats.hpp>
#include <opencv2/core/mat.hpp>
#include <opencv2/imgproc/imgproc.hpp>

using namespace std;
using namespace cv;
using namespace cvx;
using namespace crowd::bg;

Mat1d IntensityFeatureExtractor::getFeatures(Mat const& img) const
{
    Mat3b hsvImage = cvret::cvtColor(img, COLOR_BGR2HSV);

    Mat1d features(img.total(), 1);

    int nPixels = hsvImage.total();
    for (int i = 0; i < nPixels; ++i)
    {
        Vec3b hsv = at(hsvImage,i);
        double v = hsv[2]/255.0f;
        features(i,0) = v;
    }

    return features;
}
