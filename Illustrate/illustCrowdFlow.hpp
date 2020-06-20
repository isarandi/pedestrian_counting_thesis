#ifndef ILLUSTRATE_ILLUSTCROWDFLOW_HPP_
#define ILLUSTRATE_ILLUSTCROWDFLOW_HPP_

#include <pch.hpp>
#include <CrowdCounting/PersonLocations.hpp>

namespace crowd{ namespace illust {

auto createAnnotatedSlice(
		std::vector<crowd::LineCrossing> const& crossings,
		cv::Mat3b const& imageSlice,
		int dotRadius = 10
		) -> cv::Mat;

void lineCrossingsVideo(
        cvx::ImageLoadingIterable const& seq,
        crowd::PersonLocations const& locations,
        cvx::LineSegment const& segment,
        cvx::bpath const& outputPath);

}}

#endif /* ILLUSTRATE_ILLUSTCROWDFLOW_HPP_ */
