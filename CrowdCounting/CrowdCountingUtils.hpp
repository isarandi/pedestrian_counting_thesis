#ifndef CROWDCOUNTINGUTILS_HPP
#define CROWDCOUNTINGUTILS_HPP

#include <cvextra/core.hpp>
#include <CrowdCounting/RegionCounting/CountingFrame.hpp>
#include <opencv2/core/core.hpp>
#include <string>
#include <vector>

namespace crowd {

auto gridQuantize(
        std::vector<cv::Point2d> const& peoplePositions,
        cv::Size gridSize) -> std::vector<double>;

auto gridQuantizeAll(
        std::vector<CountingFrame> const& countingFrames,
        cv::Size gridSize) -> std::vector<std::vector<double> > ;

auto readPeoplePositions(std::string const& filePath)
    -> std::vector<std::vector<cv::Point2d>>;

auto generateCountingIllustration(
        cv::Mat image,
        std::vector<double> const& peopleCountsDesired,
        std::vector<double> const& peopleCountsPrediction,
        cv::Size gridSize) -> cv::Mat;

auto createGridRectangles(cv::Size gridSize)
    -> std::vector<cvx::Rectd>;

}

#endif // CROWDCOUNTINGUTILS_HPP
