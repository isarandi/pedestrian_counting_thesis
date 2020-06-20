#ifndef VIDEOITERABLE_HPP
#define VIDEOITERABLE_HPP

#include <opencv2/opencv.hpp>
#include <cvextra/filesystem.hpp>

namespace cvx {

class VideoIterator
{
public:
    VideoIterator(cv::VideoCapture* videoCapture, bool end);

    auto operator  *() -> cv::Mat&;
    auto operator ->() -> cv::Mat*;
    auto operator ++() -> VideoIterator&;
    auto operator ++(int) -> VideoIterator;
    auto operator +=(int incr) -> VideoIterator&;
    auto operator ==(VideoIterator const& rhs) -> bool;
    auto operator !=(VideoIterator const& rhs) -> bool;

private:
    static int const VIDEO_END = -1;

    int iFrame;
    cv::Mat m;
    cv::VideoCapture* videoCapture;
};

class VideoIterable
{
public:
    VideoIterable(cv::VideoCapture vidCap);
    VideoIterable(cvx::bpath const& filePath);

    auto begin() -> VideoIterator;
    auto end() -> VideoIterator;

private:
    cv::VideoCapture vidCap;

};

inline
auto video(cvx::bpath const& filePath) -> VideoIterable
{
    return VideoIterable{filePath};
}

}

#endif // VIDEOITERABLE_HPP
