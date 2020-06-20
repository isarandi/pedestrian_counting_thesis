#include <CrowdCounting/LineCounting/Features/Textons/TextureDescriptor.hpp>
#include <cvextra/coords.hpp>
#include <cvextra/LoopRange.hpp>
#include <opencv2/core/mat.hpp>
#include <opencv2/core/operations.hpp>

using namespace std;
using namespace cv;
using namespace cvx;
using namespace crowd;

auto LocalHOGDescriptor::
describePoints(
        cv::InputArray img_,
        cv::InputArray stencil_,
        OutputArray outFeatures
        ) -> vector<Point>
{
    vector<float> hogFeatures;

    Mat img = img_.getMat();
    BinaryMat stencil = stencil_.getMat();

    vector<Point> points;
    Mat outMat = outFeatures.getMat();

    for (Point p : cvx::points(img.size()))
    {
        if ((stencil.empty() || stencil(p)) && cvx::contains(img, Rect{p-cellSize, p+cellSize}))
        {
            hogCalculator.compute(img(Rect{p-cellSize, p+cellSize}), hogFeatures, cellSize*2, Size{0,0});
            Mat1f(Mat1f(hogFeatures).t()).copyTo(outMat.row(points.size()));

            points.push_back(p);
        }
    }

    return points;
}

auto LocalHOGDescriptor::
getDescriptorSize() const -> int
{
    return hogCalculator.getDescriptorSize();
}

auto FilterBankDescriptor::
describePoints(
        cv::InputArray img_,
        cv::InputArray stencil_,
        cv::OutputArray outFeatures
        ) -> vector<Point>
{
    Mat img = img_.getMat();
    Mat stencil = (stencil_.empty() ? Mat1b{} : stencil_.getMat());

    filterResponses.resize(filterBank.getFeatureCount());
    filterBank.getResponses(img, filterResponses);

    vector<Point> points;
    Mat1f outMat = outFeatures.getMat();

    int nFilters = filterBank.getFeatureCount();

    for (Point p : cvx::points(img.size()))
    {
        if (stencil_.empty() || stencil.at<uchar>(p))
        {
            for (int iFilter : cvx::irange(nFilters))
            {
                outMat(points.size(), iFilter) = filterResponses[iFilter].at<float>(p);
            }

            points.push_back(p);
        }
    }

    return points;
}

auto FilterBankDescriptor::
getDescriptorSize() const -> int
{
    return filterBank.getFeatureCount();
}

auto FilterBankDescriptor::
describe() const -> boost::property_tree::ptree
{
    boost::property_tree::ptree pt;
    pt.put("type", "FilterBankDescriptor");
    pt.put("lmSize", lmSize);
    return pt;
}

auto FilterBankDescriptor::
create(boost::property_tree::ptree const& pt) -> std::unique_ptr<FilterBankDescriptor>
{
    return stdx::make_unique<FilterBankDescriptor>(pt.get<int>("lmSize"));
}


auto LocalHOGDescriptor::
describe() const -> boost::property_tree::ptree
{
    boost::property_tree::ptree pt;
    pt.put("type", "FilterBankDescriptor");
    pt.put("cellSize.width", cellSize.width);
    pt.put("cellSize.height", cellSize.height);
    pt.put("nGradientBins", nGradientBins);
    return pt;
}

auto LocalHOGDescriptor::
create(boost::property_tree::ptree const& pt) -> std::unique_ptr<LocalHOGDescriptor>
{
    return stdx::make_unique<LocalHOGDescriptor>(
            Size{pt.get<int>("cellSize.width"), pt.get<int>("cellSize.height")},
            pt.get<int>("nGradientBins")
            );
}

auto TextureDescriptor::
create(
        boost::property_tree::ptree const& pt
) -> std::unique_ptr<TextureDescriptor>
{
    std::string type = pt.get<std::string>("type");

    if (type == "FilterBankDescriptor")
    {
        return FilterBankDescriptor::create(pt);
    }
    else if (type == "LocalHOGDescriptor")
    {
        return LocalHOGDescriptor::create(pt);
    }
    throw 1;
}
