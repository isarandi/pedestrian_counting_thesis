#include <CrowdCounting/LineCounting/LineCountingSet.hpp>
#include <CrowdCounting/LineCounting/FeatureSlices.hpp>
#include <CrowdCounting/LineCounting/WindowBasedCounting.hpp>
#include <CrowdCounting/OverallLineCounting/OverallLineCounter.hpp>
#include <Flow/flowSlice.hpp>
#include <MachineLearning/LearningSet.hpp>
#include <Run/config.hpp>

#include <Persistence.hpp>

#include <cvextra/cvret.hpp>
#include <cvextra/io.hpp>
#include <cvextra/LoopRange.hpp>
#include <cvextra/mats.hpp>
#include <cvextra/strings.hpp>
#include <cvextra/vectors.hpp>
#include <cvextra/volumetric.hpp>
#include <stdx/stdx.hpp>
#include <opencv2/core/core.hpp>
#include <vector>
#include <limits>

using namespace std;
using namespace cv;
using namespace cvx;
using namespace crowd;
using namespace crowd::linecounting;
using namespace crowd::lineopticalflow;

auto LineCountingSet::
loadLocations() const -> PersonLocations
{
    return PersonLocations{config::locationPath(datasetName)}
        .applyStencil(config::roiStencil(datasetName))
        .betweenFrames(frameRange);
}

auto LineCountingSet::
loadSlices() const -> FeatureSlices
{
	return FeatureSlices::loadOrCompute(datasetName, segment)(frameRange, Range::all());
}

auto LineCountingSet::
cross(
        vector<string> const& datasetNames,
        const vector<LineSegment>& lineSegments,
		Range const& range
        )  -> vector<LineCountingSet>
{
    std::vector<LineCountingSet> result;

    for (auto const& datasetName : datasetNames)
    {
        for (auto const& lineSegment : lineSegments)
        {
            result.push_back(
                    LineCountingSet{datasetName, lineSegment, range});
        }
    }

    return result;
}

auto LineCountingSet::
verticalSub(cv::Range const& subRange) const -> LineCountingSet
{
    cv::Range newRange = (frameRange == cv::Range::all()) ?
            subRange :
            cv::Range{frameRange.start+subRange.start, frameRange.start+subRange.end};

    if (!cvx::contains(frameRange, newRange))
    {
        throw "Doesn't contain";
    }

    return {datasetName, segment, newRange};
}

auto LineCountingSet::
parallelLineSets(
        std::vector<OverallLineCountingSet> const& overallSets,
        int nLines
        ) -> std::vector<LineCountingSet>
{
    vector<LineSegment> lines;
    Size size = config::frames(overallSets[0].datasetName)[0].size();

    for (int ix : cvx::irange(nLines))
    {
        double x = (1+ix) * size.width/(nLines+1);
        lines.push_back({{x,0},{x, size.height-1}});
    }

    vector<LineCountingSet> lineCountingSets;
    for (auto const& o : overallSets)
    {
        cvx::vectors::push_back_all(lineCountingSets, LineCountingSet::cross({o.datasetName}, lines, o.frameRange));
    }

    return lineCountingSets;
}


auto LineCountingSet::
describe() const -> boost::property_tree::ptree
{
	boost::property_tree::ptree pt;
	pt.put("datasetName", datasetName);
	pt.put("frame_start", frameRange.start);
	pt.put("frame_end", frameRange.end);
	pt.put("segment_p1_x", segment.p1.x);
	pt.put("segment_p1_y", segment.p1.y);
	pt.put("segment_p2_x", segment.p2.x);
	pt.put("segment_p2_y", segment.p2.y);
	return pt;
}

auto LineCountingSet::
create(boost::property_tree::ptree const& pt) -> std::unique_ptr<LineCountingSet>
{
	return stdx::make_unique<LineCountingSet>(
			pt.get<string>("datasetName"),
			LineSegment{
				{pt.get<double>("segment_p1_x"),pt.get<double>("segment_p1_y")},
				{pt.get<double>("segment_p2_x"),pt.get<double>("segment_p2_y")}},
			Range(pt.get<int>("frame_start"),pt.get<int>("frame_end"))
	);
}

