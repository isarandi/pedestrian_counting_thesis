/*
 * flowSlice.hpp
 *
 *  Created on: Oct 27, 2014
 *      Author: sarandi
 */

#ifndef FLOW_FLOWSLICE_HPP_
#define FLOW_FLOWSLICE_HPP_

#include "lineFlow.hpp"
#include <opencv2/core/core.hpp>
#include <cvextra/ImageLoadingIterable.hpp>
#include <cvextra/math.hpp>

namespace crowd {

namespace lineopticalflow {

template <typename ImageSequence>
auto createFlowSlice(
		ImageSequence images,
        cvx::LineSegment const& segment,
        int T,
        int indexOffset,
        OpticalFlowOptions const& options
        ) -> cv::Mat2d;


auto createFlowSliceParallel(
		cvx::ImageLoadingIterable const& images,
        cvx::LineSegment const& segment,
        OpticalFlowOptions const& options
        ) -> cv::Mat2d;

}
}

#endif /* FLOW_FLOWSLICE_HPP_ */
