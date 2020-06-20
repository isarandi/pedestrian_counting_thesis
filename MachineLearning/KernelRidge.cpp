#include <cvextra/cvret.hpp>
#include <cvextra/LoopRange.hpp>
#include <cvextra/vectors.hpp>
#include <cvextra/math.hpp>
#include <MachineLearning/KernelRidge.hpp>
#include <MachineLearning/LearningSet.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/core/mat.hpp>
//#include <Eigen/Core>
//#include <Eigen/Cholesky>
//#include <cvextra/eigen.hpp>
#include <cvextra/armadillo.hpp>
#include <algorithm>
#include <cmath>
#include <iostream>
#include <string>
#include <vector>
#include <omp.h>

using namespace std;
using namespace cv;
using namespace cvx;
using namespace crowd;

KernelRidge::
KernelRidge(KernelRidgeParams const& params, double rbfGamma, bool predictWithoutConfidence)
    : params(params)
	, kernel(stdx::make_cloned_unique<SquaredExponentialKernel>(rbfGamma))
    , predictWithoutConfidence(predictWithoutConfidence)
{}

void KernelRidge::
train(LearningSet const& ls)
{
	Mat1d inputs = ls.X;
	Mat1d outputs = ls.Y;

    // select a smaller number of random rows because of memory limitations
    int nKeptExamples = std::min(28000, inputs.rows);
    Mat1d trainingOutputs;
    if (nKeptExamples < inputs.rows)
    {
        std::srand(10); //deterministic selection
        vector<int> permutation = cvx::vectors::range(inputs.rows);
        std::random_shuffle(permutation.begin(), permutation.end());

        trainingOutputs.create(nKeptExamples, outputs.cols);
        for (Point p : cvx::points(trainingOutputs))
        {
            trainingOutputs(p) = outputs(permutation[p.y],p.x);
        }

        trainingInputs.create(nKeptExamples, inputs.cols);
        for (Point p : cvx::points(trainingInputs))
        {
            trainingInputs(p) = inputs(permutation[p.y],p.x);
        }
    } else {
        trainingInputs = inputs;
        trainingOutputs = outputs;
    }

    cout << "Calculating kernel matrix for " << trainingInputs.rows << " training examples..." << endl;
    Mat1d K = kernel->calcSymmetricKernelMatrix(trainingInputs);

    for (int iDiag : cvx::irange(K.rows))
    {
        K(iDiag,iDiag) += params.regularizationLambda;
    }

    cout << "Doing Cholesky..." << endl;
    arma::mat L = arma::chol(cvret::cv2armaTr(K));
    cout << "Cholesky decomposition done!" << endl;

    arma::mat armaA =
            arma::solve(
                    arma::trimatu(L),
                    arma::solve(
                            arma::trimatl(L.t()),
                            cvret::cv2arma(trainingOutputs)));

    cv::transpose(cvret::arma2cvTr(armaA), A);
}

auto KernelRidge::
predict(Mat1d const& testInputs) const -> Mat1d
{
    return kernel->calcKernelMatrix(testInputs, trainingInputs) * A;
}

auto KernelRidge::
predictWithConfidence(Mat1d const& testInputs) const -> PredictionWithConfidence
{
    if (predictWithoutConfidence)
    {
        Mat1d mean = predict(testInputs);
        return {mean, Mat1d::ones(mean.size())};
    }

    Mat1d K_testTrain = kernel->calcKernelMatrix(testInputs, trainingInputs);

    arma::mat armaV = arma::solve(arma::trimatl(L.t()), cvret::cv2armaTr(K_testTrain));
    Mat1d V = cvret::arma2cvTr(armaV);
    Mat1d varianceReduction = cvret::reduce(cvx::sq(V), 1, REDUCE_SUM);

    Mat1d K_testTest = kernel->calcDiagonalOfKernelMatrix(testInputs);
    Mat1d variances = params.variance_n + params.variance_f*(K_testTest - varianceReduction);

    return {K_testTrain * A, cv::repeat(variances, 1, A.cols)};
}

auto KernelRidge::
getJacobian(Mat1d const& testInputs) const -> Mat1d
{
    int sizes[] = {testInputs.rows, testInputs.cols, A.cols};
    Mat1d grad{3, sizes};

    Mat1d K_testTrain = kernel->calcKernelMatrix(testInputs, trainingInputs);

    for (int iOutput : cvx::irange(A.cols))
    {
        Mat1d thisOutputGrad = kernel->calcGradientOfGPMean(K_testTrain, trainingInputs, testInputs, A.col(iOutput));

        for (Point p : cvx::points(testInputs))
        {
            grad(p.y, p.x, iOutput) = thisOutputGrad(p);
        }
    }

    return grad;
}

auto KernelRidge::
getDescription() const -> string
{
    return "Kernel ridge (rbf-kernel): lambda=" +
                    std::to_string(params.regularizationLambda);
}

auto KernelRidge::
describe() const -> boost::property_tree::ptree
{
	boost::property_tree::ptree pt;
	pt.put("type", "KernelRidge");

	pt.put("variance_f", params.variance_f);
	pt.put("variance_n", params.variance_n);

	return pt;
}

auto KernelRidge::
create(boost::property_tree::ptree const& pt) -> std::unique_ptr<KernelRidge>
{
	return stdx::make_unique<KernelRidge>(
			GaussianProcessParam{pt.get<double>("variance_f"), pt.get<double>("variance_n")},
			pt.get<double>("rbfGamma")
	);
}
