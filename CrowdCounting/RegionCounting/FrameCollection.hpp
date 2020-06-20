#ifndef FRAMECOLLECTION_HPP
#define FRAMECOLLECTION_HPP

#include <CrowdCounting/RegionCounting/CountingFrame.hpp>
#include <string>
#include <vector>

namespace crowd {

/**
 * @brief A named collection of CountingFrame objects with support for
 * copying parts of it.
 */
class FrameCollection
{
public:

    FrameCollection(
            std::string const& name,
            std::vector<CountingFrame> const& frames)
        : name(name)
        , frames(frames){}

    FrameCollection(){}

    auto rangeFrom(int begin) const -> FrameCollection;
    auto range(int begin, int end) const -> FrameCollection;
    auto range(cv::Range const& range) const -> FrameCollection;
    auto rangeByRatio(double begin, double end) const -> FrameCollection;
    auto getDescription() const -> std::string;

    auto size() const -> int {return frames.size();}
    auto operator [](int i) const -> CountingFrame const& {return frames[i];}
    auto begin() const -> std::vector<CountingFrame>::const_iterator {return frames.begin();}
    auto end() const -> std::vector<CountingFrame>::const_iterator {return frames.end();}

    auto getFrames() const -> std::vector<CountingFrame> const& {return frames;}
    auto getName() const -> std::string {return name;}

    void append(FrameCollection const& other);

private:
    std::vector<CountingFrame> frames;
    std::string name;

};

}

#endif // FRAMECOLLECTION_HPP
