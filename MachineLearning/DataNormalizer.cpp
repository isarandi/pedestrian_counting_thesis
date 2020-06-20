#include <MachineLearning/DataNormalizer.hpp>
#include <Persistence.hpp>
#include <cvextra/cvenums.hpp>
#include <cvextra/cvret.hpp>
#include <cvextra/LoopRange.hpp>
#include <cvextra/mats.hpp>
#include <opencv2/core/core_c.h>
#include <opencv2/core/mat.hpp>

using namespace std;
using namespace cv;
using namespace cvx;
using namespace crowd;

auto DataNormalizer::
applyRet(cv::InputArray src) -> cv::Mat const
{
   cv::Mat result;
   apply(src, result);
   return result;
}


void PerFeatureNormalizer::
train(InputArray data)
{
    Mat mat = data.getMat();
    cv::reduce(mat, means, 0, REDUCE_AVG);
    cvx::variance(mat, stdevs, 0, -1, means);
    cv::sqrt(stdevs, stdevs);
}

void PerFeatureNormalizer::
apply(InputArray src, OutputArray dst) const
{
    Mat x = src.getMat();
    if (dst.empty())
    {
        dst.create(x.size(), x.type());
    }

    Mat result = dst.getMat();

    for (int row : cvx::irange(x.rows))
    {
        auto target = result.row(row);
        cv::subtract(x.row(row), means, target);
        cv::divide(target, stdevs, target);
    }
}

auto PerFeatureNormalizer::
getJacobian(InputArray singleInput) const -> Mat
{
    return cvx::mats::asDiagonal(1.0/stdevs);
}

void PerFeatureNormalizer::
savePersistence(std::string const& name) const
{
    Persistence::save(name+"_means", means);
    Persistence::save(name+"_stdevs", stdevs);
}

void PerFeatureNormalizer::
loadPersistence(std::string const& name)
{
    means = Persistence::loadMat(name+"_means");
    stdevs = Persistence::loadMat(name+"_stdevs");
}

////

void Whitener::
train(InputArray src)
{
    Mat x = src.getMat();
    cv::reduce(src, means, 0, REDUCE_AVG);

    Mat1d covar;
    cv::calcCovarMatrix(src, covar, means, CV_COVAR_NORMAL | CV_COVAR_ROWS | CV_COVAR_SCALE);

    Mat s, u, vt;
    cv::SVDecomp(covar, s, u, vt);

    double epsilon = 1e-7;
    weights =  u * cvx::mats::asDiagonal(1.0 / cvret::sqrt(s + epsilon)) * u.t();
}

void Whitener::
apply(InputArray src, OutputArray dst) const
{
    Mat x = src.getMat();
    Mat result = dst.getMat();

    for (int row : cvx::irange(x.rows))
    {
        auto target = result.row(row);
        cv::subtract(x.row(row), means, target);
        Mat(target * weights).copyTo(target);
    }
}

auto Whitener::
getJacobian(InputArray singleInput) const -> Mat
{
    return weights.t();
}

void Whitener::
savePersistence(std::string const& name) const
{
    Persistence::save(name+"_means", means);
    Persistence::save(name+"_weights", weights);
}

void Whitener::
loadPersistence(std::string const& name)
{
    means = Persistence::loadMat(name+"_means");
    weights = Persistence::loadMat(name+"_weights");
}


auto Whitener::
describe() const -> boost::property_tree::ptree
{
    boost::property_tree::ptree pt;
    pt.put("type", "Whitener");
    return pt;
}

auto Whitener::
create(boost::property_tree::ptree const& pt) -> std::unique_ptr<Whitener>
{
    return stdx::make_unique<Whitener>();
}

auto PerFeatureNormalizer::
describe() const -> boost::property_tree::ptree
{
    boost::property_tree::ptree pt;
    pt.put("type", "PerFeatureNormalizer");
    return pt;
}

auto PerFeatureNormalizer::
create(boost::property_tree::ptree const& pt) -> std::unique_ptr<PerFeatureNormalizer>
{
    return stdx::make_unique<PerFeatureNormalizer>();
}

auto DataNormalizer::
create(
        boost::property_tree::ptree const& pt
) -> std::unique_ptr<DataNormalizer>
{
    std::string type = pt.get<std::string>("type");

    if (type == "PerFeatureNormalizer")
    {
        return PerFeatureNormalizer::create(pt);
    }
    else if (type == "Whitener")
    {
        return Whitener::create(pt);
    }
    throw 1;
}


