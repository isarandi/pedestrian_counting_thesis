#include <cvextra/core.hpp>
#include <cvextra/mats.hpp>
#include <CrowdCounting/RegionCounting/Features/Extractors/Edges.hpp>
#include <CrowdCounting/RegionCounting/Features/Impl/features.hpp>
#include <CrowdCounting/RegionCounting/Features/PreprocessedFrame.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/core/mat.hpp>
#include <string>
#include <vector>

using namespace std;
using namespace cv;
using namespace cvx;
using namespace crowd;

std::vector<double> Edges::extract(PreprocessedFrame const& frame, Rectd relativeRect) const
{
    Mat maskPart = cvx::extractRelativeRoi(frame.mask, relativeRect);
    Mat edgesPart = cvx::extractRelativeRoi(frame.edges, relativeRect);
    Mat scalePart = cvx::extractRelativeRoi(frame.scaleMap, relativeRect);

    Mat maskedEdges = edgesPart.clone();
    maskedEdges.setTo(0, maskPart==0);

    return {crowd::weightedPixelCount(maskedEdges, 1.0/scalePart)};
}

int Edges::getFeatureCount() const
{
    return 1;
}

std::vector<string> Edges::getNames() const
{
    return {"foreground Edges"};
}

string Edges::getDescription() const
{
    return "Foreground Edges";
}

auto Edges::
describe() const -> boost::property_tree::ptree
{
    boost::property_tree::ptree pt;
    pt.put("type", "Edges");
    return pt;
}

auto Edges::
create(boost::property_tree::ptree const& pt) -> std::unique_ptr<Edges>
{
    return stdx::make_unique<Edges>();
}
