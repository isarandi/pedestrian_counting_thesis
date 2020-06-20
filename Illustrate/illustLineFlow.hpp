#ifndef ILLUSTRATE_ILLUSTLINEFLOW_HPP_
#define ILLUSTRATE_ILLUSTLINEFLOW_HPP_

#include <cvextra.hpp>

namespace cvx {
namespace illust{

std::vector<cv::Mat> inspectFlow(
        cv::Mat2d const& flow,
        cv::Mat3b const& im1,
        cv::Mat3b const& im2,
        cvx::LineSegment const& segment,
		int factor = 10,
		int step = 1);

}

}


#endif /* ILLUSTRATE_ILLUSTLINEFLOW_HPP_ */
