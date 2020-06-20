#include <Camera/RealisticCameraModel.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/core/core.hpp>
#include <fstream>

using namespace std;
using namespace cv;
using namespace crowd;

RealisticCameraModel::
RealisticCameraModel(string const& xmlPath)
{
    ifstream xmlFileStream(xmlPath);
    wrappedModel.fromXml(xmlFileStream);

    resolution = Size(wrappedModel.width(), wrappedModel.height());
}

auto RealisticCameraModel::
imageToWorld(Point2d imagePoint, double worldZ) const -> Point3d
{
    double worldX;
    double worldY;

    wrappedModel.imageToWorld(imagePoint.x, imagePoint.y, worldZ, worldX, worldY);
    return Point3d(worldX, worldY, worldZ);
}

auto RealisticCameraModel::
worldToImage(Point3d worldPoint) const -> Point2d
{
    double imX;
    double imY;

    wrappedModel.worldToImage(worldPoint.x, worldPoint.y, worldPoint.z, imX, imY);
    return Point2d(imX, imY);
}
