#ifndef VOLUMETRIC_HPP
#define VOLUMETRIC_HPP

#include "ImageLoadingIterable.hpp"
#include "math.hpp"

namespace cvx {

template <typename ImageSequence>
auto timeSlice(
        ImageSequence frames,
        cvx::LineSegment const& seg
        ) -> cv::Mat;

}

#endif // VOLUMETRIC_HPP
