#ifndef FEATURES_HPP
#define FEATURES_HPP

#include <opencv2/core/core.hpp>
#include <string>
#include <vector>

namespace crowd {

auto createIndexSuffixedVersions(
		std::string const& name,
		int count
		) -> std::vector<std::string>;

auto weightedPixelCount(
		cv::InputArray src,
		cv::InputArray weights
		) -> double;

auto weightedMaxResponseHistogram(
        cv::InputArray input,
        cv::InputArray weights,
        std::vector<cv::Mat> const& filters
		) -> std::vector<double>;

auto statisticalLandscape(
		cv::InputArray src,
		std::vector<double> const& thresholds
		) -> std::vector<double>;

auto Minkowski(
		cv::InputArray src,
		int maxRadius = 15
		) -> double;

auto GLCMFeatures(
		cv::InputArray in,
		cv::InputArray mask
		) -> std::vector<double>;


}

#endif // FEATURES_HPP
