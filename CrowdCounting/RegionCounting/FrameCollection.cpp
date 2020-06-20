#include <cvextra/vectors.hpp>
#include <cvextra/LoopRange.hpp>
#include <CrowdCounting/RegionCounting/FrameCollection.hpp>
#include <opencv2/core/core.hpp>
#include <stdx/stdx.hpp>
#include <boost/range.hpp>
#include <sstream>

using namespace std;
using namespace cv;
using namespace cvx;
using namespace crowd;

FrameCollection FrameCollection::
rangeFrom(int begin) const
{
    return this->range(begin, this->size());
}

FrameCollection FrameCollection::
range(int begin, int end) const
{
    vector<CountingFrame> framesRange = cvx::vectors::subVector(frames, begin, end);

    stringstream ss;
    ss << name << " (" << begin << " .. "<< end << ")";
    string nameRange = ss.str();

    return FrameCollection(nameRange, framesRange);
}

FrameCollection FrameCollection::
rangeByRatio(double begin, double end) const
{
    vector<CountingFrame> framesRange = cvx::vectors::subVectorByRatio(frames, begin, end);

    stringstream ss;
    ss << name << " (" << stdx::to_string(begin, 2) << " .. "<< stdx::to_string(end, 2) << ")";
    string nameRange = ss.str();

    return FrameCollection(nameRange, framesRange);
}

auto crowd::FrameCollection::
range(const cv::Range& range) const -> FrameCollection
{
    return (range == Range::all()) ? *this : this->range(range.start, range.end);
}

auto FrameCollection::
getDescription() const -> string
{
    bool allHaveSameProcessingSize =
            std::all_of(
                frames.begin(), frames.end(),
                [&](CountingFrame const& f){
                    return f.getProcessingSize()==frames[0].getProcessingSize();});

    if (allHaveSameProcessingSize)
    {
        Size resolution = frames[0].getProcessingSize();
        stringstream ss;
        ss << name << " w=" << resolution.width << " h=" << resolution.height;
        return ss.str();
    } else {
        stringstream ss;
        ss << name << " mixed resolution";
        return ss.str();
    }
}

void crowd::FrameCollection::
append(const FrameCollection& other)
{
    name += " + " + other.name;
    cvx::vectors::push_back_all(frames, other.frames);
}
