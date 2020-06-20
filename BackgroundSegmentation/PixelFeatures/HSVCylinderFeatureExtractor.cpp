#include <BackgroundSegmentation/PixelFeatures/HSVCylinderFeatureExtractor.hpp>
#include <cvextra/cvret.hpp>
#include <cvextra/mats.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/core/mat.hpp>
#include <opencv2/core/types_c.h>
#include <opencv2/imgproc/imgproc.hpp>
#include <cmath>

using namespace std;
using namespace cv;
using namespace cvx;
using namespace crowd::bg;

Mat1d HSVCylinderFeatureExtractor::getFeatures(Mat const& img) const
{
    Mat1d features{(int)img.total(), 5};

    Mat3b hsvImage = cvret::cvtColor(img, COLOR_BGR2HSV);
    Mat1b grayImg = cvret::cvtColor(img, COLOR_BGR2GRAY);

    Mat1s gradX = cvret::Sobel(grayImg, CV_16S, 1, 0);
    Mat1s gradY = cvret::Sobel(grayImg, CV_16S, 0, 1);

    int nPixels = hsvImage.total();

    #pragma omp parallel for
    for (int i = 0; i < nPixels; ++i)
    {
        Vec3b const& hsv = at(hsvImage, i);
        double h = hsv[0] / 180.0;
        double s = hsv[1] / 255.0;
        double v = hsv[2] / 255.0;

        double factor = s * v;
        double x = std::cos(2*CV_PI*h) * factor;
        double y = std::sin(2*CV_PI*h) * factor;

        double gx = at(gradX,i)/255.0;
        double gy = at(gradY,i)/255.0;

        cvx::mats::setRow(features, i, {x, y, v*scaleOfV, gx*0.5, gy*0.5});
    }

    return features;
}
