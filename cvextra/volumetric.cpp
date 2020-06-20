#include "volumetric.hpp"
#include "improc.hpp"
#include "mats.hpp"
#include "ImageLoadingIterable.hpp"

using namespace std;
using namespace cv;
using namespace cvx;

template <typename ImageSequence>
auto cvx::
timeSlice(
        ImageSequence frames,
        LineSegment const& seg
        ) -> Mat
{
    int nSamplesAlongLine = (int)(seg.properties().length);

    cv::Mat result{0,nSamplesAlongLine, (*frames.begin()).type()};
    for (cv::Mat const& frame : frames)
    {
        cvx::vconcat(result, cvxret::lineProfile(frame, seg), result);
    }
    return result;
}

// instantiate
#define INSTANTIATE_TIME_SLICE(type) \
    template \
    auto cvx::timeSlice<type>( \
            type frames, \
            LineSegment const& seg \
            ) -> Mat;

INSTANTIATE_TIME_SLICE(ImageLoadingIterable)
INSTANTIATE_TIME_SLICE(std::vector<cv::Mat>)
INSTANTIATE_TIME_SLICE(std::vector<cv::Mat3b>)
INSTANTIATE_TIME_SLICE(std::vector<cv::Mat1b>)
INSTANTIATE_TIME_SLICE(Iterable<typename ImageLoadingIterable::iterator_t>)
