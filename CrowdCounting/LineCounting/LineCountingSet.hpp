#ifndef CROWDCOUNTING_LineCounting_LINECOUNTINGSET1_HPP_
#define CROWDCOUNTING_LineCounting_LINECOUNTINGSET1_HPP_

#include <cvextra/math.hpp>
#include <CrowdCounting/LineCounting/SlidingWindow.hpp>
#include <CrowdCounting/LineCounting/WindowBasedCounting.hpp>
#include <CrowdCounting/PersonLocations.hpp>
#include <cvextra/configfile.hpp>
#include <opencv2/core/core.hpp>
#include <string>

namespace crowd{ namespace linecounting {

class OverallLineCountingSet;

class LineCountingSet {
public:
    std::string datasetName;
    cvx::LineSegment segment;
    cv::Range frameRange;

    auto verticalSub(cv::Range const& subRange) const -> LineCountingSet;

    auto loadLocations() const -> PersonLocations;
    auto loadSlices() const -> FeatureSlices;

    static
    auto cross(
            std::vector<std::string> const& datasetNames,
            std::vector<cvx::LineSegment> const& lineSegments,
			cv::Range const& range = cv::Range::all()
            ) -> std::vector<LineCountingSet>;

    static
    auto parallelLineSets(
            std::vector<OverallLineCountingSet> const& overallSets,
            int nLines
            ) -> std::vector<LineCountingSet>;

    CVX_CONFIG_SINGLE(LineCountingSet)

};

}} /* namespace crowd */

#endif /* CROWDCOUNTING_LineCounting_LINECOUNTINGSET1_HPP_ */
