#ifndef LOCALBINARYPATTERNSIMPL_HPP
#define LOCALBINARYPATTERNSIMPL_HPP

#include <opencv2/core/core.hpp>
#include <vector>

namespace crowd {

auto localBinaryPatternHistograms(
        cv::InputArray input,
        double radius,
        int nSamples,
        int nHistogramBins,
        cv::Size subGrid)
-> std::vector<double>;


namespace localbinarypatterns {

auto localDescriptor(cv::InputArray src, cv::Point location, std::vector<cv::Vec2i> const& diffs) -> int;
auto getHistogram(
        cv::InputArray input,
        std::vector<cv::Vec2i> const& diffs,
        int nHistogramBins)
-> std::vector<double>;

void drawAll();


} //namespace localbinarypatterns

} //namespace crowd
#endif // LOCALBINARYPATTERNSIMPL_HPP
