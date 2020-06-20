#ifndef REALISTICCAMERAMODEL_HPP
#define REALISTICCAMERAMODEL_HPP

#include <Camera/AbstractCameraModel.hpp>
#include <Camera/cameraModel.hpp>
#include <opencv2/core/core.hpp>
#include <string>

namespace crowd {

/**
 * @brief Wrapper around the Etiseo::CameraModel class
 */
class RealisticCameraModel : public AbstractCameraModel
{
public:
    RealisticCameraModel(std::string const& xmlPath);

    // AbstractCameraModel interface
    virtual auto imageToWorld(cv::Point2d imagePoint, double worldZ) const -> cv::Point3d;
    virtual auto worldToImage(cv::Point3d worldPoint) const -> cv::Point2d;

private:
    Etiseo::CameraModel wrappedModel;
};

}

#endif // REALISTICCAMERAMODEL_HPP
