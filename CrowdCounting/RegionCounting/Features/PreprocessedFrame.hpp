#ifndef LOADEDCOUNTINGFRAME_HPP
#define LOADEDCOUNTINGFRAME_HPP

#include <cvextra/core.hpp>
#include <opencv2/core/core.hpp>

namespace crowd
{
class CountingFrame;
} /* namespace crowd */

namespace crowd {

/**
 * @brief Contains data that is used by several feature extractors.
 * The frame is first converted to this format and then passed into the
 * FeatureExtractor.
 */
class PreprocessedFrame
{
public:
    PreprocessedFrame(CountingFrame const& countingFrame);

    cv::Mat3b colorFrame;
    cv::Mat1b grayFrame;

    cvx::BinaryMat mask;
    cv::Mat1d scaleMap;
    cvx::BinaryMat edges;
    cv::Mat1b textonMap;
};



}

#endif // LOADEDCOUNTINGFRAME_HPP
