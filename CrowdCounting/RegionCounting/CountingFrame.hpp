#ifndef COUNTINGFRAME_HPP
#define COUNTINGFRAME_HPP

#include <cvextra/core.hpp>
#include <cvextra/filesystem.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/core/mat.hpp>
#include <string>
#include <vector>

namespace crowd {

/**
 * @brief Encapsulates a frame, its foreground mask, scale map and people ground truth locations.
 * Implemented with lazy loading (stores file paths, opens when getter function called).
 */
class CountingFrame
{
public:
    CountingFrame(
            cvx::bpath const& framePath,
            cvx::bpath const& maskPath,
			cvx::bpath const& textonMapPath,
            std::vector<cv::Point2d> const& peoplePositions,
            cv::Size processingSize,
            cv::Mat const& scaleMap,
            cvx::BinaryMat const& roi);

    auto getFrame() const -> cv::Mat;
    auto getMask() const -> cvx::BinaryMat;
    auto getScaleMap() const -> cv::Mat1d;
    auto getTextonMap() const -> cv::Mat1b;

    auto getFullResolutionFrame() const -> cv::Mat;

    auto getPeoplePositions() const -> std::vector<cv::Point2d>;
    auto getProcessingSize() const -> cv::Size;

    auto countPeopleInRectangle(cvx::Rectd relativeRectangle) const -> double;

    void saveVariant(std::string const& variantName, cv::InputArray variant) const;

    static auto allFromFolder(
            cvx::bpath const& folderPath,
            cvx::bpath const& maskPath,
			cvx::bpath const& textonMapPath,
            std::vector<std::vector<cv::Point2d>> peoplePositions,
            cv::Size processingSize,
            cv::Mat scaleMap,
            cvx::BinaryMat roi = cv::Mat())
    -> std::vector<CountingFrame>;

private:
    cvx::bpath framePath;
    cvx::bpath maskPath;
    cvx::bpath textonMapPath;
    std::vector<cv::Point2d> peoplePositions;

    cv::Size processingSize;
    cv::Mat1d scaleMap;
    cvx::BinaryMat roiMask;
    cvx::bpath parentPath;

};

}

#endif // COUNTINGFRAME_HPP
