#include <cvextra.hpp>

#include <MachineLearning/LearningSet.hpp>
#include <MachineLearning/Ridge.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/core/mat.hpp>
//#include <cvextra/eigen.hpp>
#include <cvextra/armadillo.hpp>
#include <string>
#include <vector>

using namespace std;
using namespace cv;
using namespace cvx;
using namespace crowd;


Ridge::
Ridge(double C)
    : ridgeC(C)
    , weights(stdx::make_cloned_unique<Mat1d>())
	, weightPriorVariance(1)
{}

Ridge::
Ridge(double C, double weightPriorVariance)
    : ridgeC(C)
    , weights(stdx::make_cloned_unique<Mat1d>())
	, weightPriorVariance(weightPriorVariance)
{}


void Ridge::
train(LearningSet const& ls)
{
    cvx::mats::checkNaN(ls.X, "X");
    cvx::mats::checkNaN(ls.Y, "Y");

    Mat1d appendedX = cvret::hconcat(Mat1d::ones(ls.X.rows, 1), ls.X);
    Mat1d regularizedCovariance = appendedX.t() * appendedX;

    double lambda = (1.0 / (2.0 * ridgeC));
    for (int iDiag : cvx::irange(1,regularizedCovariance.rows))
    {
        regularizedCovariance(iDiag,iDiag) += lambda;
    }

    cvx::mats::checkNaN(regularizedCovariance, "regularizedCovariance");

    //Eigen::MatrixXd eigenRegCov = cvret::cv2eigen(regularizedCovariance);
    //llt = stdx::make_copy_constructed_unique<EigenLLT>(eigenRegCov.llt());

    //Mat1d rightSide = appendedX.t()*ls.Y;

    L = arma::chol(cvret::cv2armaTr(regularizedCovariance));

    // following is armaA = solve(regularizedCovariance, appendedX.t()*ls.Y);
    arma::mat armaA =
            arma::solve(
                    arma::trimatu(L),
                    arma::solve(
                            arma::trimatl(L.t()),
                            cvret::cv2armaTr(appendedX) * cvret::cv2arma(ls.Y)));

    cv::transpose(cvret::arma2cvTr(armaA), *weights);

    //Eigen::MatrixXd eigenRightSide = cvret::cv2eigen(rightSide);
    //Eigen::MatrixXd eigenWeights = llt->solve(eigenRightSide);
    //cv::eigen2cv(eigenWeights, *weights);

    cvx::mats::checkNaN(*weights, "weights");
}

auto Ridge::
predict(Mat1d const& X) const -> Mat1d
{
    Mat1d appendedX = cvret::hconcat(Mat1d::ones(X.rows,1), X);
    cvx::mats::checkNaN(appendedX, "predictAppended");

    Mat1d Y = appendedX * (*weights);
    cvx::mats::checkNaN(Y, "predictOutput");
    return Y;
}

auto Ridge::
predictWithConfidence(Mat1d const& testInputs) const -> PredictionWithConfidence
{
    Mat1d appendedX = cvret::hconcat(Mat1d::ones(testInputs.rows,1), testInputs);
    Mat1d Y = appendedX * (*weights);

    //Eigen::MatrixXd eigenAppendedX = cvret::cv2eigen(appendedX);
    //Eigen::MatrixXd eigenV = llt->matrixL().solve(eigenAppendedX.transpose());
    arma::mat armaV = arma::solve(arma::trimatl(L.t()), cvret::cv2armaTr(appendedX));

	//Mat1d V = cvret::eigen2cv(eigenV).t();
    Mat1d V = cvret::arma2cvTr(armaV);
	Mat1d varianceSecondPart  = cvret::reduce(cvx::sq(V), 1, REDUCE_SUM);

	double noiseVariance = (1/(2*ridgeC)) * weightPriorVariance;
	Mat1d varianceForSingleOutput =
			noiseVariance * (Mat1d::ones(testInputs.rows, 1) + varianceSecondPart);

	return {Y, cv::repeat(varianceForSingleOutput, 1, weights->cols)};
}

auto Ridge::
getJacobian(cv::Mat1d const& X) const -> cv::Mat1d
{
    return weights->t();
}

auto Ridge::
getDescription() const -> string
{
    return "Linear ridge: C=" + std::to_string(ridgeC);
}

auto Ridge::
describe() const -> boost::property_tree::ptree
{
	boost::property_tree::ptree pt;
	pt.put("type", "Ridge");
	pt.put("ridgeC", ridgeC);
	pt.put("weightPriorVariance", weightPriorVariance);

	return pt;
}

auto Ridge::
create(boost::property_tree::ptree const& pt)  -> std::unique_ptr<Ridge>
{
	return stdx::make_unique<Ridge>(pt.get<double>("ridgeC"), pt.get<double>("weightPriorVariance"));
}
