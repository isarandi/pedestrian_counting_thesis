#include <cvextra/LoopRange.hpp>
#include <Camera/AbstractCameraModel.hpp>
#include <opencv2/core/mat.hpp>
#include <opencv2/core/core.hpp>

using namespace cv;
using namespace cvx;
using namespace crowd;

auto AbstractCameraModel::
renderScaleMap(Size outputSize) const -> Mat1d
{
    Mat1d mat(outputSize);

    double factor = resolution.height/(double)outputSize.height;

    for (Point const& p : cvx::points(mat))
    {
        double pixelsPerMeter = 1./factor * getPixelsPerMeter(Point2d(p.x,p.y)*factor);
        mat(p) = pixelsPerMeter;
    }

    return mat;
}

auto AbstractCameraModel::
getPixelsPerMeter(Point2d p) const -> double
{
    Point3d worldPointMiddle = imageToWorld(p, 1700./2.); //assume point over ground by 85 cm (half human)
    Point3d worldPointBottom = {worldPointMiddle.x, worldPointMiddle.y, 0.};
    Point3d worldPointTop = {worldPointMiddle.x, worldPointMiddle.y, 1700.};

    Point2d imagePointBottom = worldToImage(worldPointBottom);
    Point2d imagePointTop = worldToImage(worldPointTop);

    double lengthInPx = cv::norm(imagePointBottom-imagePointTop);

    return lengthInPx/1.7;
}
