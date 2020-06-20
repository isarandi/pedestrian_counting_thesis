#ifndef PERSON_LOCATIONS_H
#define PERSON_LOCATIONS_H

#include <cvextra/math.hpp>
#include <cvextra/filesystem.hpp>
#include <opencv2/core/core.hpp>
#include <vector>

namespace cvx
{
class LineSegment;
}

namespace crowd {

class PersonInstance
{
public:
    int personId;
    int frameId;
    cv::Point2d pos;
    cvx::Size2d size;
    bool visible;
};

enum class CrossingDir {
    LEFTWARD = 1,
    RIGHTWARD = 2,
    STANDING = 3,
};

auto opposite(CrossingDir dir) -> CrossingDir;
auto dir01(CrossingDir dir) -> int;

class LineCrossing {
public:
    int personId;
    int frameId;
    double localPos;
    cv::Point2d globalPos;
    CrossingDir dir;
    bool visible;
};

class PersonLocations
{
public:
    PersonLocations(): nFrames(0){}
    explicit PersonLocations(int nFrames): nFrames(nFrames){}
    explicit PersonLocations(cvx::bpath const& filepath);

    auto getGroupedByPerson() const -> std::vector<std::vector<PersonInstance>>;
    auto getGroupedByFrame() const -> std::vector<std::vector<PersonInstance>>;
    auto getInstances() const -> std::vector<PersonInstance> {return instances;}

    auto getRelativePositionsByFrame(cv::Size size) const -> std::vector<std::vector<cv::Point2d>>;
    auto getLineCrossings(cvx::LineSegment const& seg) const -> std::vector<LineCrossing>;

    auto getInstantFlow(
            cvx::LineSegment const& seg,
            double radius = 1e-3,
            int cancellationDistance = 1
            ) const -> cv::Mat1d;

    auto simpleInstantFlow(
            cvx::LineSegment const& seg
            ) const -> cv::Mat1d;

    auto getFrameRange() const -> cv::Range;
    auto getFrameCount() const -> int;

    void add(PersonLocations const& others, int frameOffset, bool newPeople=true);

    auto betweenFrames(cv::Range range) const -> PersonLocations;
    auto cropRoi(cv::Rect const& roi) const -> PersonLocations;
    auto rescalePositions(double scalingFactor) const -> PersonLocations;
    auto timeStretchInterpolate(int newFrameCount) const -> PersonLocations;
    auto applyStencil(cvx::BinaryMat const& stencil) const -> PersonLocations;

    void writeToFile(cvx::bpath const& p);

    static auto
    fromLineAnnotations(cvx::bpath const& p) -> PersonLocations;

private:

    std::vector<PersonInstance> instances;
    int nFrames;
};

}

#endif // PERSON_LOCATIONS_H
