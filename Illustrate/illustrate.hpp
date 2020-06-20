#ifndef ILLUSTRATE_HPP
#define ILLUSTRATE_HPP


#include "Illustrate/illustCrowdFlow.hpp"
#include "Illustrate/illustLineFlow.hpp"

auto illustrateComponents(cvx::BinaryMat const& img) -> cv::Mat3b;
auto illustrateLabels(cv::Mat const& img, int nLabels) -> cv::Mat3b;

#endif // ILLUSTRATE_HPP
