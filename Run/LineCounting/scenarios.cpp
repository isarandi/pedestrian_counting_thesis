#include "scenarios.hpp"
#include <boost/range/irange.hpp>
#include <cvextra/LoopRange.hpp>
#include <cvextra/math.hpp>
#include <CrowdCounting/LineCounting/LineCountingSet.hpp>
#include <opencv2/core/core.hpp>
#include <CrowdCounting/LineCounting/LineCountingExperiment.hpp>
#include <string>
#include <vector>

using namespace std;
using namespace cv;
using namespace cvx;
using namespace crowd;
using namespace crowd::linecounting;

auto crowd::
getStandardLines(cv::Size size, int nLines) -> vector<LineSegment>
{
    vector<LineSegment> lines;
    for (int ix : cvx::irange(nLines))
    {
    	double x = (1+ix) * size.width/(nLines+1);
        lines.push_back({{x,0},{x, size.height-1}});
    }
    return lines;
}

auto crowd::
getLongSequenceScenario() -> LineCountingScenario
{
    string dataSetName = "crange_ausschnitt1";
    auto lines = getStandardLines({960,540}, 7);

    LineCountingScenario scenario;
    scenario.trainings = LineCountingSet::cross({dataSetName}, lines, {0,2000});
    scenario.tests = LineCountingSet::cross({dataSetName}, lines, {2000,3000});
    return scenario;
}

auto crowd::
getDensitiesToLongScenario() -> LineCountingScenario
{
    LineCountingScenario scenario;
    vector<string> trainingDatasetNames = {
            "crange_densities/12",
            "crange_densities/13",
            "crange_densities/21",
            "crange_densities/22",
            "crange_densities/23",
            "crange_densities/24",
            "crange_densities/31",
            "crange_densities/32",
            "crange_densities/33",
            "crange_densities/34",
            "crange_densities/41",
            "crange_densities/42",
            "crange_densities/43",
            "crange_densities/44",
            };
    auto lines = getStandardLines({960,540},7);
    scenario.trainings = LineCountingSet::cross(trainingDatasetNames, lines);

    string testDatasetName =  "crange_ausschnitt1";
    scenario.tests = LineCountingSet::cross({testDatasetName}, lines, Range{0,2000});

    return scenario;
}

auto crowd::
getViddScenario() -> LineCountingScenario
{
	string dataSetName = "ucsd_vidd";
    auto lines = getStandardLines(Size{238,158},5);

	LineCountingScenario scenario;
    scenario.trainings = LineCountingSet::cross({dataSetName}, lines, {0,2000});
    scenario.tests = LineCountingSet::cross({dataSetName}, lines, {2000,3000});
    return scenario;
}

auto crowd::
getSmallScenario() -> LineCountingScenario
{

    vector<string> trainingDatasetNames = {
//            "crange_densities/12",
//            "crange_densities/13",
            "crange_densities/21",
            "crange_densities/22",
            "crange_densities/24",
//            "crange_densities/24",
//            "crange_densities/31",
//            "crange_densities/32",
//            "crange_densities/33",
//            "crange_densities/34",
//            "crange_densities/41",
//            "crange_densities/42",
//            "crange_densities/43",
//            "crange_densities/44",
            };

    auto lines = getStandardLines({960,540},7);

    string testDatasetName = "crange_densities/23";

    LineCountingScenario scenario;
    scenario.trainings = LineCountingSet::cross(trainingDatasetNames, lines);
    scenario.tests = LineCountingSet::cross({testDatasetName}, lines);
    return scenario;
}

auto crowd::
getMiniScenario() -> LineCountingScenario
{
    vector<string> trainingDatasetNames = {
//            "crange_densities/12",
//            "crange_densities/13",
            "crange_densities/21",
//            "crange_densities/24",
//            "crange_densities/31",
//            "crange_densities/32",
//            "crange_densities/33",
//            "crange_densities/34",
//            "crange_densities/41",
//            "crange_densities/42",
//            "crange_densities/43",
//            "crange_densities/44",
            };
    string testDatasetName = "crange_densities/23";

    auto lines = getStandardLines({960,540},2);
    LineCountingScenario scenario;
    scenario.trainings = LineCountingSet::cross(trainingDatasetNames, lines);
    scenario.tests = LineCountingSet::cross({testDatasetName}, lines);

    return scenario;
}

auto crowd::
getSmallCrangeScenario() -> OverallLineCountingScenario
{
    vector<string> trainingDatasetNames = {
           "crange_densities/12",
    };

    vector<string> testDatasetNames = {
           "crange_densities/22",
    };

    return OverallLineCountingScenario{
        OverallLineCountingSet::makeMany(trainingDatasetNames),
        OverallLineCountingSet::makeMany(testDatasetNames)};
}

auto crowd::
getCrangeLineValidationScenario() -> OverallLineCountingScenario
{
    vector<string> trainingDatasetNames = {
           "crange_densities/12",
           "crange_densities/13",
           "crange_densities/21",
           "crange_densities/23",
           "crange_densities/24",
           "crange_densities/31",
           "crange_densities/32",
           "crange_densities/33",
           "crange_densities/41",
           "crange_densities/42",
           "crange_densities/44",
    };

    vector<string> testDatasetNames = {
           "crange_densities/22",
           "crange_densities/34",
           "crange_densities/43",
    };

    return OverallLineCountingScenario{
        OverallLineCountingSet::makeMany(trainingDatasetNames, {8,251}),
        OverallLineCountingSet::makeMany(testDatasetNames, {8,251})};
}

auto crowd::
getCrangeLineTestScenario() -> OverallLineCountingScenario
{
    vector<string> trainingDatasetNames = {
           "crange_densities/12",
           "crange_densities/13",
           "crange_densities/21",
           "crange_densities/23",
           "crange_densities/24",
           "crange_densities/31",
           "crange_densities/32",
           "crange_densities/33",
           "crange_densities/41",
           "crange_densities/42",
           "crange_densities/44",
           "crange_densities/22",
           "crange_densities/34",
           "crange_densities/43",
    };

    return OverallLineCountingScenario{
        OverallLineCountingSet::makeMany(trainingDatasetNames),//, {8,251}),
        OverallLineCountingSet::makeMany({"crange_ausschnitt1"})};//, {8,4501})};
}

auto crowd::
getCrangeUltimateLineTestScenario() -> OverallLineCountingScenario
{
    vector<string> trainingDatasetNames = {
           "crange_densities/12",
           "crange_densities/13",
           "crange_densities/21",
           "crange_densities/23",
           "crange_densities/24",
           "crange_densities/31",
           "crange_densities/32",
           "crange_densities/33",
           "crange_densities/41",
           "crange_densities/42",
           "crange_densities/44",
           "crange_densities/22",
           "crange_densities/34",
           "crange_densities/43",
           "crange_ausschnitt1"
    };

    return OverallLineCountingScenario{
        OverallLineCountingSet::makeMany(trainingDatasetNames),//, {8,251}),
        OverallLineCountingSet::makeMany({"crange_long"})};//, {8,4501})};
}

auto crowd::
getViddValidationScenario() -> OverallLineCountingScenario
{
    return OverallLineCountingScenario{
        OverallLineCountingSet::makeMany({"ucsd_vidd1_orig"},Range{0,6000}),
        OverallLineCountingSet::makeMany({"ucsd_vidd1_orig"},Range{6000,9000})};
}

auto crowd::
getViddSmallScenario() -> OverallLineCountingScenario
{
    return OverallLineCountingScenario{
        OverallLineCountingSet::makeMany({"ucsd_vidd1_orig"},Range{0,20}),
        OverallLineCountingSet::makeMany({"ucsd_vidd1_orig"},Range{20,50})};
}

auto crowd::
getViddTestScenario() -> OverallLineCountingScenario
{
    return OverallLineCountingScenario{
        OverallLineCountingSet::makeMany({"ucsd_vidd1_orig"},Range{0,9000}),
        OverallLineCountingSet::makeMany({"ucsd_vidd1_orig"},Range{9000,12000})};
}
