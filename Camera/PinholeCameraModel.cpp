#include <cvextra/core.hpp>
#include <Camera/PinholeCameraModel.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/core/types_c.h>
#include <cmath>

using namespace cv;
using namespace cvx;
using namespace crowd;

PinholeCameraModel::
PinholeCameraModel(
        double cameraAltitudeInMillimeters,
        double lookdownAngleInRadians,
        double focalLengthInMillimeters,
        Size resolution)
{
    focalLength = focalLengthInMillimeters;
    this->resolution = resolution;
    opticalCenter = 0.5 * Point2d(resolution);

    double rotationAngle = CV_PI/2-lookdownAngleInRadians;
    double cosBeta = std::cos(rotationAngle);
    double sinBeta = std::sin(rotationAngle);

    cameraRotation = Matx33d(1,0,0, 0,cosBeta,sinBeta, 0,-sinBeta,cosBeta);
    cameraRotationInv = Matx33d(1,0,0, 0,cosBeta,-sinBeta, 0,sinBeta,cosBeta);
    cameraTranslation = Vec3d(0, 0, -cameraAltitudeInMillimeters);
}

auto PinholeCameraModel::
imageToWorld(Point2d imagePoint, double worldZ) const -> Point3d
{
    Point3d possiblePointFromCamera = Point3d(
                (imagePoint.x-opticalCenter.x)/focalLength*(-1.0),
                (imagePoint.y-opticalCenter.y)/focalLength*(-1.0), -1.0);

    Point3d possiblePointWithCameraAtOrigin = cameraRotationInv * possiblePointFromCamera;

    double desiredZWithCameraAtOrigin = worldZ + cameraTranslation[2];
    double scaleFactor = desiredZWithCameraAtOrigin / possiblePointWithCameraAtOrigin.z;

    Point3d desiredPointWithCameraAtOrigin = possiblePointWithCameraAtOrigin*scaleFactor;
    Point3d desiredWorldPoint = desiredPointWithCameraAtOrigin - cameraTranslation;

    return desiredWorldPoint;
}

auto PinholeCameraModel::
worldToImage(Point3d worldPoint) const -> Point2d
{
    Point3d pointFromCamera = cameraRotation * (worldPoint+cameraTranslation);
    Point2d pointFromCameraXY = Point2d(pointFromCamera.x, pointFromCamera.y);

    return pointFromCameraXY * focalLength / (-pointFromCamera.z) + opticalCenter;
}
