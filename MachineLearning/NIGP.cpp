#include <MachineLearning/NIGP.hpp>
#include <Python/Pyplot.hpp>
#include <MachineLearning/LearningSet.hpp>
#include <cvextra/armadillo.hpp>
#include <cvextra/cvret.hpp>
#include <cvextra/cvenums.hpp>
#include <cvextra/LoopRange.hpp>
#include <cvextra/vectors.hpp>
#include <cvextra/math.hpp>
#include <cvextra/io.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/core/mat.hpp>
//#include <Eigen/Core>
//#include <Eigen/Cholesky>
//#include <cvextra/eigen.hpp>
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <armadillo>
#include <string>
#include <vector>
#include <omp.h>

using namespace std;
using namespace cv;
using namespace cvx;
using namespace crowd;

NIGP::
NIGP(KernelRidgeParams const& params, double rbfGamma, double inputNoise, bool predictWithoutConfidence, int nIterations)
    : params(params)
    , kernel(stdx::make_cloned_unique<SquaredExponentialKernel>(rbfGamma))
	, inputNoiseVarianceConstant(inputNoise) //0.014
    , predictWithoutConfidence(predictWithoutConfidence)
    , nIterations(nIterations)
{
}

void NIGP::
train(LearningSet const& ls)
{
    Mat1d inputs = ls.X;
    Mat1d outputs = ls.Y;
    //llts.resize(outputs.cols);
    Ls.resize(outputs.cols);
    inputNoiseVariances = Mat1d{1, inputs.cols, inputNoiseVarianceConstant};

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

    cout<< "Calculating kernel matrix for " << nKeptExamples << " training examples out of " << inputs.rows << " total" << endl;

    Mat1d K = kernel->calcSymmetricKernelMatrix(trainingInputs);

    // add regularization before solving for A
    for (int iDiag : cvx::irange(K.rows))
    {
        K(iDiag,iDiag) += params.regularizationLambda;
    }

    cout<< "Solving for A..." << endl;
    A.create(trainingOutputs.size());
	solveForA(K, trainingOutputs, -1);

    // remove regul
    for (int iDiag : cvx::irange(K.rows))
    {
        K(iDiag,iDiag) = 1.0;
    }

    for (int iOutput : cvx::irange(trainingOutputs.cols))
    {
        for (int iIter : cvx::irange(nIterations))
        {
            cout<< "Solving for gradient-based increment..." << endl;
            Mat1d gradientBasedRegularizationIncrements;
            {
                Mat1d grad = kernel->calcGradientOfGPMean(K, trainingInputs, trainingInputs, A.col(iOutput));
                gradientBasedRegularizationIncrements = cvx::sq(grad) * inputNoiseVariances.t() / params.variance_f;
            }

            // add regul before solving for A
            for (int iDiag : cvx::irange(K.rows))
            {
                K(iDiag,iDiag) = 1.0 + gradientBasedRegularizationIncrements(iDiag,0) + params.regularizationLambda;
            }

        	cout<< "Solving for A..." << endl;
        	solveForA(K, trainingOutputs, iOutput);

        	// remove regul
            for (int iDiag : cvx::irange(K.rows))
            {
                K(iDiag,iDiag) -= params.regularizationLambda;
            }

        }

        if (iOutput < trainingOutputs.cols-1)
        {
            // restore original
            for (int iDiag : cvx::irange(K.rows))
            {
                K(iDiag,iDiag) = 1.0;
            }
        }
    }
    cout << "NIGP training done!" << endl;
}

auto NIGP::
predict(Mat1d const& testInputs) const -> Mat1d
{
    return kernel->calcKernelMatrix(testInputs, trainingInputs) * A;
}

auto NIGP::
predictWithConfidence(Mat1d const& testInputs) const -> PredictionWithConfidence
{
    if (predictWithoutConfidence)
    {
        Mat1d mean = predict(testInputs);
        return {mean, Mat1d::ones(mean.size())};
    }


	cout << "NIGP prediction..." << endl;

	Mat1d K_testTrain = kernel->calcKernelMatrix(testInputs, trainingInputs);
	//Eigen::MatrixXd eigenTestTrain = cvret::cv2eigen(K_testTrain);

	Mat1d varianceReduction{testInputs.rows, A.cols};
	Mat1d varianceIncrement{testInputs.rows, A.cols};

	for (int iOutput : cvx::irange(A.cols))
	{
		//Eigen::MatrixXd eigenV = llts[iOutput]->matrixL().solve(eigenTestTrain.transpose());
		//Mat1d V = cvret::eigen2cv(eigenV).t();
	    arma::mat armaV = arma::solve(arma::trimatl(Ls[iOutput].t()), cvret::cv2armaTr(K_testTrain));
	    Mat1d V = cvret::arma2cvTr(armaV);
		cv::reduce(cvx::sq(V), varianceReduction.col(iOutput), 1, REDUCE_SUM);

		Mat1d grad = kernel->calcGradientOfGPMean(K_testTrain, trainingInputs, testInputs, A.col(iOutput));
		varianceIncrement.col(iOutput) = cvx::sq(grad)*inputNoiseVariances.t();
	}

	Mat1d K_testTest = kernel->calcDiagonalOfKernelMatrix(testInputs);

	cout << "F" << endl;
	Mat1d variance =
			params.variance_n + varianceIncrement +
			params.variance_f*(cv::repeat(K_testTest, 1, A.cols)-varianceReduction);
	cout << "G" << endl;

	//    pyx::Pyplot plt;
	//    plt.plot(params.variance_f*varianceReduction, "g");
	//    plt.plot(varianceIncrement.col(0), "r");
	//    plt.plot(varianceIncrement.col(1), "r--");
	//    plt.plot(params.variance_f*K_testTest, "k");
	//    plt.plot(variance.col(0), "b");
	//    plt.plot(variance.col(1), "b--");
	//    plt.show();

	return {K_testTrain * A, variance};
}

auto NIGP::
getJacobian(Mat1d const& testInput) const -> Mat1d
{
    assert(testInput.rows==1);

    Mat1d grad{A.cols, trainingInputs.cols};

    Mat1d K_testTrain = kernel->calcKernelMatrix(testInput, trainingInputs);

    for (int iOutput : cvx::irange(A.cols))
    {
        Mat1d thisOutputGrad = kernel->calcGradientOfGPMean(K_testTrain, trainingInputs, testInput, A.col(iOutput));
        thisOutputGrad.copyTo(grad.row(iOutput));
    }

    return grad;
}

void NIGP::
solveForA(
        Mat1d const& regularizedK,
        Mat1d const& trainingOutputs,
        int iOutput)
{
    //auto eigenRegK = cvret::cv2eigen(regularizedK);
    //auto llt = eigenRegK.llt();

    arma::mat L = arma::chol(cvret::cv2armaTr(regularizedK));
    cout << "Cholesky decomposition done!" << endl;

    if (iOutput != -1)
    {
        //llts[iOutput] = stdx::make_copy_constructed_unique<EigenLLT>(llt);
        Ls[iOutput] = L;
    } else if (nIterations==0)
    {
        for (int i : cvx::irange(Ls.size()))
        {
            //llts[i] = stdx::make_copy_constructed_unique<EigenLLT>(llt);
            Ls[i] = L;
        }
    }

    cv::Range colRange = (iOutput == -1 ? cv::Range::all() : cv::Range{iOutput, iOutput+1});

    //auto eigenOutputs = cvret::cv2eigen(trainingOutputs.colRange(colRange));
    //Eigen::MatrixXd es = llt.matrixL().solve(eigenOutputs);
   // Eigen::MatrixXd eigenA = llt.matrixL().transpose().solve(es);

    arma::mat armaA =
            arma::solve(
                    arma::trimatu(L),
                    arma::solve(
                            arma::trimatl(L.t()),
                            cvret::cv2arma(trainingOutputs.colRange(colRange))));

    //cv::eigen2cv(eigenA, stdx::tempref(A.colRange(colRange)));

    cv::transpose(cvret::arma2cvTr(armaA), A.colRange(colRange));
}

auto NIGP::
getDescription() const -> string
{
    return "Kernel ridge (rbf-kernel): lambda=" +
                    std::to_string(params.regularizationLambda);
}

auto NIGP::
describe() const -> boost::property_tree::ptree
{
	boost::property_tree::ptree pt;
	pt.put("type", "NIGP");

	pt.put("variance_f", params.variance_f);
	pt.put("variance_n", params.variance_n);

	pt.put("inputNoiseFactor", inputNoiseVarianceConstant);

	return pt;
}

auto NIGP::
create(boost::property_tree::ptree const& pt) -> std::unique_ptr<NIGP>
{
	return stdx::make_unique<NIGP>(
			GaussianProcessParam{pt.get<double>("variance_f"), pt.get<double>("variance_n")},
			pt.get<double>("rbfGamma"),
			pt.get<double>("inputNoiseFactor")
	);
}
