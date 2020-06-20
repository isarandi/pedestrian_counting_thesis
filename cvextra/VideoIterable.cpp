#include "VideoIterable.hpp"

using namespace cvx;

VideoIterator::VideoIterator(cv::VideoCapture* _videoCapture, bool end)
    : videoCapture(_videoCapture)
{
    if (end)
    {
        iFrame = VideoIterator::VIDEO_END;
    } else
    {
        *videoCapture >> m;
        iFrame = 0;
    }
}

cv::Mat& VideoIterator::operator *()
{
    return m;
}

cv::Mat* VideoIterator::operator ->()
{
    return &m;
}

VideoIterator& VideoIterator::operator ++()
{
    *videoCapture >> m;

    if (m.empty())
    {
        iFrame = VideoIterator::VIDEO_END;
    } else
    {
        iFrame++;
    }

    return *this;
}

VideoIterator VideoIterator::operator ++(int)
{
    VideoIterator tmp(*this);
    ++(*this);
    return tmp;
}

VideoIterator& VideoIterator::operator +=(int incr)
{
    for (int i=0; i < incr-1; i++)
    {
        videoCapture->grab();
    }
    return ++(*this);
}

bool VideoIterator::operator ==(VideoIterator const& rhs)
{
    return iFrame == rhs.iFrame
            && videoCapture == rhs.videoCapture;
}
bool VideoIterator::operator !=(VideoIterator const& rhs)
{
    return !(*this == rhs);
}

VideoIterable::VideoIterable(cv::VideoCapture vidCap)
    : vidCap(vidCap){}

VideoIterable::VideoIterable(cvx::bpath const& filePath)
    : vidCap(filePath.string()){}

VideoIterator VideoIterable::begin()
{
    return VideoIterator(&vidCap, false);
}

VideoIterator VideoIterable::end()
{
    return VideoIterator(&vidCap, true);
}
