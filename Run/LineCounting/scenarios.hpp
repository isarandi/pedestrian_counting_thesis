#ifndef RUN_LINECOUNTING_SCENARIOS_H_
#define RUN_LINECOUNTING_SCENARIOS_H_

#include <vector>
#include <cvextra/math.hpp>
#include <CrowdCounting/LineCounting/LineCountingExperiment.hpp>
#include <CrowdCounting/OverallLineCounting/OverallLineCounter.hpp>

namespace crowd {

auto getStandardLines(cv::Size size = {960,540}, int nLines = 7) -> std::vector<cvx::LineSegment>;
auto getTestLines() -> std::vector<cvx::LineSegment>;
auto getLongSequenceScenario() -> crowd::linecounting::LineCountingScenario;
auto getDensitiesToLongScenario() -> crowd::linecounting::LineCountingScenario;
auto getViddScenario() -> crowd::linecounting::LineCountingScenario;
auto getSmallScenario() -> crowd::linecounting::LineCountingScenario;
auto getMiniScenario() -> crowd::linecounting::LineCountingScenario;

auto getCrangeLineValidationScenario() -> crowd::linecounting::OverallLineCountingScenario;
auto getCrangeLineTestScenario() -> crowd::linecounting::OverallLineCountingScenario;
auto getCrangeUltimateLineTestScenario() -> crowd::linecounting::OverallLineCountingScenario;
auto getViddValidationScenario() -> crowd::linecounting::OverallLineCountingScenario;
auto getViddTestScenario() -> crowd::linecounting::OverallLineCountingScenario;

auto getViddSmallScenario() -> crowd::linecounting::OverallLineCountingScenario;


auto getSmallCrangeScenario() -> crowd::linecounting::OverallLineCountingScenario;

}

#endif /* RUN_LINECOUNTING_SCENARIOS_H_ */
