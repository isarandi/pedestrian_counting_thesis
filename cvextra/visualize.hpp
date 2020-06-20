#ifndef VISUALIZE_HPP
#define VISUALIZE_HPP

#include "core.hpp"

namespace cvx {
namespace visu
{

auto vectorFieldAsHSV(cv::Mat2d const& field, double saturationMagnitude=30) -> cv::Mat3b;
auto vectorFieldAsHSVAsBGR(cv::Mat2d const& field, double saturationMagnitude=30) -> cv::Mat3b;
auto vectorFieldAsArrows(cv::Mat const& background, cv::Mat2d const& field, int step, double minMagnitude) -> cv::Mat;
auto grayStretch(cv::InputArray src) -> cv::Mat1d;

void maskIllustration(cv::InputArray src, cv::InputArray mask, cv::OutputArray illust);
auto maskIllustration(cv::InputArray src, cv::InputArray mask) -> cv::Mat;

void highlightOnto(cv::InputArray orig, cv::InputArray mask, cv::InputOutputArray illust, cv::Scalar color);
auto darkened(cv::InputArray orig) -> cv::Mat3b;

auto components(cvx::BinaryMat const& img) -> cv::Mat3b;
auto labels(cv::Mat const& img) -> cv::Mat3b;

}
}



#endif // VISUALIZE_HPP
