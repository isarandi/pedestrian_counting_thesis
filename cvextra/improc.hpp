#ifndef IMAGEUTILS_HPP
#define IMAGEUTILS_HPP

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include "core.hpp"
#include "math.hpp"
#include "LoopRange.hpp"
#include "filesystem.hpp"

namespace cvx {

enum class AngleUnit {
    DEGREES,
    RADIANS,
};

auto getGaussianKernel(cv::Size ksize, double sigmaX, double sigmaY, int ktype=CV_64F) -> cv::Mat;
void resizeBest(cv::InputArray src, cv::OutputArray dst, cv::Size desiredSize, double fx=0, double fy=0);

void rotate(
        cv::InputArray src,
        cv::OutputArray dst,
        cv::Point2d anchor,
        double angle,
        AngleUnit angleUnit = AngleUnit::DEGREES,
        int flags = cv::INTER_LINEAR,
        int borderMode = cv::BORDER_DEFAULT,
        cv::Scalar borderValue = cv::Scalar(0));

void gradientField(cv::InputArray src1d, cv::OutputArray dst2d);
void setChannel(cv::InputArray src, cv::InputOutputArray dst, int channelIdx);

void putTextCentered(
        cv::Mat& targetImage,
        std::string const& text,
        cv::Point2d centerAnchor,
        int fontFace,
        double fontScale,
        cv::Scalar textColor);

void putTextCairo(
        cv::Mat &targetImage,
        std::string const& text,
        cv::Point2d centerAnchor,
        std::string const& fontFamily,
        double fontSize,
        cv::Scalar textColor,
        bool fontItalic = false,
        bool fontBold = false);

void grid(cv::Mat& dst, cv::Size gridResolution, cv::Scalar color, int thickness = 1);
void arrow(cv::Mat& dst, cv::Point2d start, cv::Vec2d direction, cv::Scalar color, int thickness = 1);



/**
 * @brief Wrapper around cv::findContours but this one does not change the input image.
 */
void findContours(cv::InputArray image, cv::OutputArrayOfArrays dst, int mode, int method, cv::Point offset=cv::Point());

int connectedComponents(cv::InputArray src, cv::OutputArray dst, int connectivity, int ddepth);

template<typename T>
auto bilinear(
        cv::InputArray src,
        cv::Point2d location,
        int borderType = cv::BORDER_DEFAULT
        ) -> T;

template<typename T, int D>
auto bilinear(
        cv::InputArray src,
        cv::Point2d location,
        int borderType = cv::BORDER_DEFAULT
        ) -> cv::Vec<T,D>;

template <typename T, int D>
void lineProfile(
        cv::InputArray src,
        cv::OutputArray dst,
        cvx::LineSegment const& segment);

void lineProfile(
        cv::InputArray src,
        cv::OutputArray dst,
        cvx::LineSegment const& segment);

auto boundingBoxOfBinary(cv::InputArray src) -> cv::Rect;

int const FILL = -1;

enum class Connectivity {
    EIGHT,
    FOUR,
};

} // namespace cvx

namespace cvxret
{
    auto resizeBest(cv::InputArray src, cv::Size desiredSize, double fx=0, double fy=0) -> cv::Mat;
    auto draw(cv::Mat const& mat, cvx::BinaryMat const& stencil, cv::Scalar color) -> cv::Mat;
    auto resizeByRatio(cv::InputArray src, double factor, int interpolation = -1) -> cv::Mat;
    auto rotate(
            cv::InputArray src,
            cv::Point2d anchor,
            double angle,
            cvx::AngleUnit angleUnit = cvx::AngleUnit::DEGREES,
            int flags = cv::INTER_LINEAR,
            int borderMode = cv::BORDER_DEFAULT,
            cv::Scalar borderValue = cv::Scalar(0)) -> cv::Mat;
    auto gradientField(cv::Mat1d const& mat) -> cv::Mat2d;
    auto skeleton(cv::InputArray src) -> cvx::BinaryMat;

    auto lineProfile(cv::InputArray src, cvx::LineSegment const& seg) -> cv::Mat;

    auto findContours(cv::InputArray image, int mode, int method, cv::Point offset=cv::Point()
            ) -> std::vector<std::vector<cv::Point>>;

    auto removeSmallConnectedComponents(
            cvx::BinaryMat const& binary,
            int minSizeToKeep,
            cvx::Connectivity connectivity = cvx::Connectivity::FOUR
            ) -> cvx::BinaryMat;

    auto fillSmallHoles(
            cvx::BinaryMat const& binary,
            int minSizeToKeep,
            cvx::Connectivity connectivity = cvx::Connectivity::FOUR
            ) -> cvx::BinaryMat;
    auto outline(cv::InputArray binary) -> cvx::BinaryMat;

    auto cannyPlus(
    		cv::InputArray const& src,
			double gaussRadius,
			double threshold1,
			double threshold2,
			int minComponentSize
			)->cvx::BinaryMat;

}

#endif // IMAGEUTILS_HPP
