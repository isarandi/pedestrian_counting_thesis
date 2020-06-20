#include <boost/algorithm/string/predicate.hpp>
#include <cvextra/coords.hpp>
#include <cvextra/filesystem.hpp>
#include <cvextra/io.hpp>
#include <cvextra/LoopRange.hpp>
#include <cvextra/math.hpp>
#include <cvextra/cvret.hpp>
#include <cvextra/mats.hpp>
#include <cvextra/strings.hpp>
#include <Python/Pyplot.hpp>
#include <stdx/stdx.hpp>
#include <CrowdCounting/PersonLocations.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/core/core.hpp>
#include <algorithm>
#include <map>
#include <limits>
#include <string>
#include <vector>

using namespace std;
using namespace cv;
using namespace cvx;
using namespace crowd;


PersonLocations::
PersonLocations(bpath const& filePath)
{
    for (auto line : cvx::io::linesOf(filePath))
    {
        if (boost::algorithm::starts_with(line, "#"))
            continue;

        auto parts = cvx::str::split(line, " ");

        if (parts.size()==1)
        {
            nFrames = std::stoi(parts[0]);
        }

        if (parts.size()==7)
        {
            PersonInstance pi;
            pi.personId = std::stoi(parts[0]);
            pi.frameId =  std::stoi(parts[1]);
            pi.pos.x = std::stod(parts[2]);
            pi.pos.y = std::stod(parts[3]);
            pi.size.width = std::stod(parts[4]);
            pi.size.height = std::stod(parts[5]);
            pi.visible = (std::stoi(parts[6]) == 1);
            instances.push_back(pi);
        }
    }

}

auto PersonLocations::
getGroupedByPerson() const -> vector<vector<PersonInstance> >
{
    vector<vector<PersonInstance>> result;
    map<int, int> personIdToIndex;

    for (PersonInstance pi : instances)
    {
        if (personIdToIndex.count(pi.personId) == 0)
        {
            result.emplace_back();
            personIdToIndex[pi.personId] = result.size()-1;
        }
        int index = personIdToIndex[pi.personId];
        result[index].push_back(pi);
    }

    auto comparator =
            [](PersonInstance const& pi1,
               PersonInstance const& pi2){return pi1.frameId < pi2.frameId;};

    // Sort each trajectory by time
    for (auto& traj : result)
    {
        std::sort(traj.begin(), traj.end(), comparator);
    }

    return result;
}

auto PersonLocations::
getGroupedByFrame() const -> vector<vector<PersonInstance> >
{
    vector<vector<PersonInstance>> result;

    for (PersonInstance pi : instances)
    {
        while (result.size() < pi.frameId+1)
        {
            result.push_back({});
        }
        result[pi.frameId].push_back(pi);
    }

    while (result.size() < nFrames)
    {
        result.push_back({});
    }

    return result;
}

auto PersonLocations::
getLineCrossings(LineSegment const& countingSegment) const -> vector<LineCrossing>
{
    auto trajectories = getGroupedByPerson();
    vector<LineCrossing> crossings;
    LineSegmentProperties countingSegProp = countingSegment.properties();

    for (auto const& traj : trajectories)
    {
        auto trajSegmentStart = traj[0].pos;
        auto prevFrameId = traj[0].frameId;

        for (auto personInstance : traj)
        {
            if (personInstance.frameId != prevFrameId+1)
            {
                prevFrameId = personInstance.frameId;
                continue;
            }

            auto trajSegmentEnd = personInstance.pos;

            auto trajectorySegment = LineSegment{trajSegmentStart, trajSegmentEnd};
            auto intersectResult = LineSegment::intersect(countingSegment, trajectorySegment);

            if (intersectResult.intersects)
            {
                auto cDir = countingSegProp.dir;
                auto tDir = trajectorySegment.dir();

                auto crossProd = Vec3d{cDir[0], cDir[1], 0.0}.cross({tDir[0], tDir[1], 0.0});
                auto direction = (crossProd[2] > 0) ? CrossingDir::LEFTWARD : CrossingDir::RIGHTWARD;

                auto localIntersectionPos = countingSegProp.globalToLocal(intersectResult.intersectionPoint);

                auto crossing = LineCrossing{
                    personInstance.personId,
                    personInstance.frameId,
                    localIntersectionPos,
                    intersectResult.intersectionPoint,
                    direction,
                    personInstance.visible};

                crossings.push_back(crossing);
            }

            trajSegmentStart = trajSegmentEnd;
            prevFrameId = personInstance.frameId;
        }
    }

return crossings;
}

auto PersonLocations::
betweenFrames(Range range) const -> PersonLocations
{
    if (range == cv::Range::all())
    {
        return *this;
    }

    PersonLocations newLocations{(int)range.size()};
    for (auto const& instance : this->instances)
    {
        if (cvx::contains(range, instance.frameId))
        {
            newLocations.instances.push_back(
                {instance.personId, instance.frameId-range.start, instance.pos, instance.size, instance.visible});
        }
    }
    return newLocations;
}


auto PersonLocations::
getFrameCount() const -> int
{
    if (nFrames != -1)
    {
        return nFrames;
    } else {
        throw "No nFrames set";
    }
}

auto PersonLocations::
rescalePositions(double scalingFactor) const -> PersonLocations
{
    PersonLocations newLocations{nFrames};
    for (auto const& instance : this->instances)
    {
        newLocations.instances.push_back(
        {instance.personId, instance.frameId, instance.pos*scalingFactor, instance.size, instance.visible});
    }
    return newLocations;
}

void crowd::PersonLocations::
add(PersonLocations const& others, int frameOffset, bool newPeople)
{
    int maxPersonId = stdx::max_element_by(
            instances.begin(),
            instances.end(),
            [](PersonInstance const& pi){return pi.personId;})->personId;

    for (PersonInstance instance : others.instances)
    {
        instance.frameId += frameOffset;
        instance.personId += maxPersonId;

        instances.push_back(instance);
    }
}

auto crowd::PersonLocations::
getFrameRange() const -> cv::Range
{
    int maxFrameId = stdx::max_element_by(
            instances.begin(),
            instances.end(),
            [](PersonInstance const& pi){return pi.frameId;})->frameId;

    int minFrameId = stdx::min_element_by(
            instances.begin(),
            instances.end(),
            [](PersonInstance const& pi){return pi.frameId;})->frameId;

    return cv::Range{minFrameId, maxFrameId+1};
}

auto crowd::PersonLocations::
applyStencil(cvx::BinaryMat const& stencil
        ) const -> PersonLocations
{
    PersonLocations newLocations{nFrames};

    std::copy_if(
            instances.begin(),
            instances.end(),
            std::back_inserter(newLocations.instances),
            [&](PersonInstance const& pi){
                Point pos2i = cvx::point2i(pi.pos);
                return cvx::contains(stencil, pos2i) && stencil(pos2i)!=0;});

    return newLocations;
}

auto crowd::
opposite(CrossingDir dir) -> CrossingDir
{
    return (dir == CrossingDir::LEFTWARD) ? CrossingDir::RIGHTWARD : CrossingDir::LEFTWARD;
}

auto netFlowOverLine(
		LineSegment const& seg,
		LineSegmentProperties const& segProp,
		Vec2d const& leftNormal,
		Point2d const& posBefore,
		Point2d const& posAfter,
		double radius) -> double
{
    double areaOnLeftBefore =
    		cvx::math::standardNormalCdf(
    				(posBefore-seg.p1).dot(leftNormal)/radius);

    double areaOnLeftAfter =
    		cvx::math::standardNormalCdf(
    				(posAfter-seg.p1).dot(leftNormal)/radius);

    double localPosBefore = segProp.globalToLocal(posBefore);
    double localPosAfter = segProp.globalToLocal(posAfter);
    double localPos = 0.5*(localPosBefore+localPosAfter);

    double myPortion =
    		cvx::math::standardNormalCdf((segProp.length-localPos)/radius)
    		- cvx::math::standardNormalCdf((0.0-localPos)/radius);

    return (areaOnLeftAfter-areaOnLeftBefore) * myPortion;
}

static
void takeOwnership(
        InputOutputArray requester,
        InputArray requests,
        InputArray ownables,
        OutputArray myShare)
{
    cv::divide(requester, requests, myShare);
    Mat reshapedMyShare = myShare.getMat().reshape(1);
    std::replace_if(
            reshapedMyShare.begin<double>(),
            reshapedMyShare.end<double>(),
            [](double d){return !(d>=0 && d <= 1);},
            0);

    cv::subtract(requester, myShare.getMat().mul(ownables), requester);

    Mat reshapedRequester = requester.getMat().reshape(1);
    std::replace_if(
            reshapedRequester.begin<double>(),
            reshapedRequester.end<double>(),
            [](double d){return !(d>=0);},
            0);
}

static
auto cancelNearbyOpposites(Mat1d const& m, int maxDistance) -> Mat1d
{
    int nFrames = m.rows;
    if (maxDistance == 0)
    {
        maxDistance = nFrames-1;
    }

    Mat1d left = cv::max(0, m);
    Mat1d right = -cv::min(0, m);

    Mat1d requests{right.size()};
    Mat1d common{right.size()};

    // right eats
    for (int distance : cvx::irange(1,maxDistance+1))
    {
        requests.setTo(0);
        requests.rowRange(distance, nFrames) += right.rowRange(0, nFrames-distance);
        requests.rowRange(0, nFrames-distance) += right.rowRange(distance, nFrames);

        cv::min(left, requests, common);
        left -= common;

        Mat myShare;
        takeOwnership(right.rowRange(0, nFrames-distance), requests.rowRange(distance, nFrames), common.rowRange(distance, nFrames), myShare);
        takeOwnership(right.rowRange(distance, nFrames), requests.rowRange(0, nFrames-distance), common.rowRange(0, nFrames-distance), myShare);
    }

//    if (!cvx::mats::equals(Mat1d(left-right), m))
//    {
//        pyx::Pyplot plt;
//        plt.plot(left-right, "g");
//        plt.plot(m, "r");
//        plt.show();
//        plt.close();
//    }

    return left - right;
}

auto PersonLocations::
simpleInstantFlow(
         cvx::LineSegment const& seg
         ) const -> cv::Mat1d
{
    Mat1d flows{nFrames, 2, 0.0};

    for (auto crossing : getLineCrossings(seg))
    {
        ++flows(crossing.frameId, crossing.dir == CrossingDir::LEFTWARD ? 0 : 1);
    }

    return flows;
}

auto PersonLocations::
getInstantFlow(
         cvx::LineSegment const& seg,
         double radius,
         int cancellationDistance
         ) const -> cv::Mat1d
{
    if (radius < 1e-4 && cancellationDistance == 0)
    {
        return simpleInstantFlow(seg);
    }


    Mat1d flows{nFrames, 2, 0.0};
    Mat leftCol = flows.col(0);
    Mat rightCol = flows.col(1);

    auto segProp = seg.properties();
    auto leftNormal = seg.clockwiseNormal();

    Mat1d thisPersonsNetFlowToLeft{flows.rows, 1};
    for (auto const& traj : getGroupedByPerson())
    {
        //Mat1d thisPersonsAreasOnLeftSide{flows.rows, 1, std::numeric_limits<double>::quiet_NaN()};
    	thisPersonsNetFlowToLeft.setTo(0);

        for (int i : cvx::irange(1, (int)traj.size()))
        {
        	if (traj[i].frameId == traj[i-1].frameId+1)
        	{
				thisPersonsNetFlowToLeft(traj[i].frameId) =
						netFlowOverLine(seg, segProp, leftNormal, traj[i-1].pos, traj[i].pos, radius);
        	}
        }

        thisPersonsNetFlowToLeft = cancelNearbyOpposites(thisPersonsNetFlowToLeft, cancellationDistance);

        cv::add(leftCol, thisPersonsNetFlowToLeft, leftCol, thisPersonsNetFlowToLeft > 0);
        cv::add(rightCol, -thisPersonsNetFlowToLeft, rightCol, thisPersonsNetFlowToLeft < 0);
    }

    return flows;
}

auto crowd::
dir01(CrossingDir dir) -> int
{
    return dir == CrossingDir::LEFTWARD ? 0 : (dir == CrossingDir::RIGHTWARD ? 1 : 2);
}

void crowd::PersonLocations::
writeToFile(cvx::bpath const& p)
{
    ofstream ofs{p.string()};
    ofs << nFrames << endl;

    for (PersonInstance const& pi: instances)
    {
        ofs <<
                cvx::str::format(
                        "%d %d %.1f %.1f %.1f %.1f %d",
                        pi.personId,
                        pi.frameId,
                        pi.pos.x,
                        pi.pos.y,
                        pi.size.width,
                        pi.size.height,
                        pi.visible ? 1 : 0) << endl;
    }

}

auto crowd::PersonLocations::
getRelativePositionsByFrame(cv::Size size) const -> vector<vector<Point2d>>
{
    vector<vector<Point2d>> relativePeoplePositions;

    for (auto const& instancesInFrame : getGroupedByFrame())
    {
        vector<Point2d> points;
        for (auto const& instance : instancesInFrame)
        {
            points.push_back(Point2d{instance.pos.x/size.width, instance.pos.y/size.height});
        }
        relativePeoplePositions.push_back(points);
    }

    return relativePeoplePositions;
}

auto crowd::PersonLocations::
timeStretchInterpolate(int newFrameCount) const -> PersonLocations
{
    PersonLocations result{newFrameCount};
    auto frameRange = Range{0,nFrames};
    double factor = newFrameCount/(double)frameRange.size();

    for (auto const& traj : getGroupedByPerson())
    {
        PersonInstance pi = traj[0];
        pi.frameId = (int)(factor*pi.frameId);
        result.instances.push_back(pi);

        for (int i : cvx::irange(1, (int)traj.size()))
        {
            if (traj[i].frameId == traj[i-1].frameId+1)
            {
                cout << traj[i].frameId << endl;
                Vec2d offset = traj[i].pos-traj[i-1].pos;

                int from = (int)(factor*traj[i-1].frameId)+1;
                int to = (int)(factor*traj[i].frameId)+1;

                for (int iNewFrame : cvx::irange(from, to))
                {
                    cout << "  " << traj[i].frameId << endl;
                    PersonInstance pi = traj[i-1];
                    pi.frameId = iNewFrame;
                    pi.pos = traj[i-1].pos + (iNewFrame-from)/(double)(to-from) * offset;
                    result.instances.push_back(pi);
                }
            } else {
                PersonInstance pi = traj[i];
                pi.frameId = (int)(factor*pi.frameId);
                result.instances.push_back(pi);
            }
        }
    }

    return result;
}

auto crowd::PersonLocations::
cropRoi(cv::Rect const& roi) const -> PersonLocations
{
    PersonLocations newLocations{nFrames};
    for (auto const& instance : this->instances)
    {
        newLocations.instances.push_back(
        {instance.personId, instance.frameId, instance.pos-cvx::point2d(roi.tl()), instance.size, instance.visible});
    }
    return newLocations;
}

auto PersonLocations::
fromLineAnnotations(cvx::bpath const& filePath) -> PersonLocations
{
    PersonLocations newLocations;
    int width;
    int id = 0;

    for (auto line : cvx::io::linesOf(filePath))
    {
        if (boost::algorithm::starts_with(line, "#"))
            continue;

        auto parts = cvx::str::split(line, " ");

        if (parts.size()==2)
        {
            newLocations.nFrames = std::stoi(parts[0]);
            width = std::stoi(parts[1]);
        }

        if (parts.size()==4)
        {
            PersonInstance pi;
            pi.personId = id++;
            pi.size = Size{1,1};
            pi.visible = true;

            PersonInstance piPrev = pi;
            pi.frameId =  std::stoi(parts[0]);
            piPrev.frameId = pi.frameId-1;

            double y = std::stod(parts[2]);
            int dir = std::stoi(parts[3]);

            if (dir == -1)
            {
                pi.pos = Point2d{1.0, y};
                piPrev.pos = Point2d{(double)width-1, y};
            } else
            {
                pi.pos = Point2d{(double)width-1, y};
                piPrev.pos = Point2d{1.0, y};
            }

            newLocations.instances.push_back(piPrev);
            newLocations.instances.push_back(pi);
        }
    }

    return newLocations;

}
