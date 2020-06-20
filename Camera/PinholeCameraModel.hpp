#ifndef PINHOLECAMERAMODEL_HPP
#define PINHOLECAMERAMODEL_HPP

#include <Camera/AbstractCameraModel.hpp>
#include <opencv2/core/core.hpp>

namespace crowd {

class PinholeCameraModel: public AbstractCameraModel
{
public:

    /**
     * Creates a horizontal lookdown camera. Assumes that the image plane in the world
     * has 1 millimeter by 1 millimeter sized pixels.
     **/
    PinholeCameraModel(
            double cameraAltitudeInMillimeters,
            double lookdownAngleInRadians,
            double focalLengthInMillimeters,
            cv::Size resolution);

    // AbstractCameraModel interface
    virtual auto imageToWorld(cv::Point2d imagePoint, double worldZ) const -> cv::Point3d;
    virtual auto worldToImage(cv::Point3d worldPoint) const -> cv::Point2d;

private:
    cv::Vec3d cameraTranslation;
    cv::Matx33d cameraRotation;
    cv::Matx33d cameraRotationInv;
    double focalLength;
    cv::Point2d opticalCenter;
};

}

#endif // PINHOLECAMERAMODEL_HPP
