#ifndef ABSTRACTCAMERAMODEL_HPP
#define ABSTRACTCAMERAMODEL_HPP

#include <opencv2/core/core.hpp>

namespace crowd {

/**
 * Represents a transformation between world and image.
 * World XY are on the ground, Z is the vertical axis pointing up, with Z=0 being the ground plane.
 * World coordinates are in millimeters, image coordinates in pixels.
 */
class AbstractCameraModel
{
public:

    /**
     * @brief Gets the 3D point in the scene that is seen in the image at the given imagePoint
     * and has a given Z coordinate in the world.
     */
    virtual auto imageToWorld(cv::Point2d imagePoint, double worldZ) const -> cv::Point3d = 0;

    /**
     * @brief Gets the location in the image where the given 3D point is projected at.
     */
    virtual auto worldToImage(cv::Point3d worldPoint) const -> cv::Point2d = 0;

    /**
     * @brief Creates an image where each pixel contains a number
     * that expresses how many pixels are used per meter on a vertical
     * pole of height 1.7 meters standing on the Z=0 ground plane,
     * whose midpoint is projected at the pixel in question
     * @param outputSize
     * @return
     */
    auto renderScaleMap(cv::Size outputSize) const -> cv::Mat1d;

    virtual ~AbstractCameraModel(){}

protected:
    cv::Size resolution;

    auto getPixelsPerMeter(cv::Point2d p) const -> double;

};

}

#endif // ABSTRACTCAMERAMODEL_HPP
