#ifndef FEATURESFORFLOW_HPP
#define FEATURESFORFLOW_HPP

#include <cvextra/core.hpp>
#include <cvextra/math.hpp>
#include <cvextra/improc.hpp>
#include <vector>

namespace crowd {
namespace lineopticalflow{
namespace details {

auto calcCartesianHueSat(cv::Mat3b const& bgrImage) -> cv::Mat2d;
auto calcHueSat(cv::Mat3b const& bgrImage) -> cv::Mat2d;
auto calcNormRGB(cv::Mat3b const& bgrImage) -> cv::Mat3d;
auto calcRGB(cv::Mat3b const& bgrImage) -> cv::Mat3d;
auto calcRGBAndGrad(cv::Mat3b const& bgrImage) -> cv::Mat;
auto calcCIEab(cv::Mat3b const& bgrImage) -> cv::Mat2d;
auto calcGray(cv::Mat3b const& bgrImage) -> cv::Mat1d;

}}}

#endif // FEATURESFORFLOW_HPP
