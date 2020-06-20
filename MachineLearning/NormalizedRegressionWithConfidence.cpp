#include <MachineLearning/NormalizedRegressionWithConfidence.hpp>
#include <boost/range/irange.hpp>
#include <cvextra/cvenums.hpp>
#include <cvextra/cvret.hpp>
#include <cvextra/LoopRange.hpp>
#include <MachineLearning/LearningSet.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/core/mat.hpp>
#include <opencv2/core/types_c.h>
#include <string>

using namespace std;
using namespace cv;
using namespace cvx;
using namespace crowd;

void NormalizedRegressionWithConfidence::
train(LearningSet const& ls)
{
    xNormalizer->train(ls.X);
    cv::reduce(ls.Y, ymeans, 0, REDUCE_AVG);

    innerRegression->train({
            xNormalizer->applyRet(ls.X),
            ls.Y - cv::repeat(ymeans, ls.Y.rows, 1)});
}

auto NormalizedRegressionWithConfidence::
predict(Mat1d const& X) const -> Mat1d
{
    return cv::repeat(ymeans, X.rows, 1) + innerRegression->predict(xNormalizer->applyRet(X));
}

auto NormalizedRegressionWithConfidence::
predictWithConfidence(Mat1d const& X) const -> PredictionWithConfidence
{
    PredictionWithConfidence innerResult =
            innerRegression->predictWithConfidence(xNormalizer->applyRet(X));
    innerResult.mean += cv::repeat(ymeans, X.rows, 1);

    return innerResult;
}

auto NormalizedRegressionWithConfidence::
getJacobian(Mat1d const& testInput) const -> Mat1d
{
    // chain rule
    return innerRegression->getJacobian(xNormalizer->applyRet(testInput)) *
            xNormalizer->getJacobian(testInput);
}

auto NormalizedRegressionWithConfidence::
getDescription() const -> string
{
    return "Normalized "+innerRegression->getDescription();
}

auto NormalizedRegressionWithConfidence::
describe() const -> boost::property_tree::ptree
{
	boost::property_tree::ptree pt;
	pt.put("type", "NormalizedRegressionWithConfidence");
	pt.put_child("innerRegression", innerRegression->describe());
	return pt;
}

auto NormalizedRegressionWithConfidence::
create(boost::property_tree::ptree const& pt) -> std::unique_ptr<NormalizedRegressionWithConfidence>
{
	return stdx::make_unique<NormalizedRegressionWithConfidence>(
			*RegressionWithConfidence::create(pt.get_child("innerRegression"))
	);
}

