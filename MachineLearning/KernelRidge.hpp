#ifndef KERNELRIDGE_HPP
#define KERNELRIDGE_HPP

#include <MachineLearning/LearningSet.hpp>
#include <MachineLearning/Regression.hpp>
#include <MachineLearning/Kernel.hpp>
#include <opencv2/core/core.hpp>
#include <cvextra/configfile.hpp>
#include <stdx/cloning.hpp>
//#include <Eigen/Core>
//#include <Eigen/Cholesky>
#include <armadillo>
#include <memory>
#include <string>
#include <vector>

namespace crowd {

class GaussianProcessParam
{
public:
    double signalVarianceMultiplier; // sigma_f_sq; priorVariance(f(X))==signalVarianceMultiplier if K(x,x)==1;
    double observationNoiseVariance; // sigma_sq
};

class ProbabilisticFeatureRidgeParam
{
public:
    double weightPriorVariance; // 1/alpha
    double observationNoiseVariance;
};

class RegularizedLeastSquaresParam
{
public:
    double dataFitTradeoffParameterC; // ridgeC
    double signalVarianceMultiplier;
};

struct KernelRidgeParams
{
    KernelRidgeParams(RegularizedLeastSquaresParam p)
    {
        regularizationLambda = 1/(2*p.dataFitTradeoffParameterC);
        variance_f = p.signalVarianceMultiplier;
        variance_n = regularizationLambda*variance_f;
    }

    KernelRidgeParams(ProbabilisticFeatureRidgeParam p)
    {
        variance_f = p.weightPriorVariance;
        regularizationLambda = p.observationNoiseVariance/variance_f;
        variance_n = p.observationNoiseVariance;
    }

    KernelRidgeParams(GaussianProcessParam p)
    {
        variance_f = p.signalVarianceMultiplier;
        regularizationLambda = p.observationNoiseVariance/variance_f;
        variance_n = p.observationNoiseVariance;
    }

    double regularizationLambda; // sigma_sq/beta = 1/(2C)

    double variance_n; // sigma_sq/beta = 1/(2C)

    // variance factor, so beta*Kernel is the real covariance of the underlying function
    // can be thought of as a kernel multiplier

    double variance_f;
};

class KernelRidge : public RegressionWithConfidence
{
public:
    KernelRidge(
    		KernelRidgeParams const& params,
			double rbfGamma,
			bool predictWithoutConfidence = false);

    // Regression interface
    virtual	void train(LearningSet const& ls);
    virtual	auto predict(cv::Mat1d const& X) const -> cv::Mat1d;
    virtual	auto predictWithConfidence(cv::Mat1d const& X) const -> PredictionWithConfidence;

    virtual auto getJacobian(cv::Mat1d const& X) const -> cv::Mat1d;

    virtual	auto getDescription() const -> std::string;

    CVX_CLONE_IN_DERIVED(KernelRidge)
    CVX_CONFIG_DERIVED(KernelRidge)

private:
    KernelRidgeParams params;
    stdx::cloned_unique_ptr<Kernel> kernel;

    //typedef decltype(Eigen::MatrixXd().llt()) EigenLLT;
    //stdx::copy_constructed_unique_ptr<EigenLLT> llt; // = Cholesky((K+.)^-1)
    arma::mat L;

    cv::Mat1d A; // = (K+.)^-1 * Y
    cv::Mat1d trainingInputs;

    bool predictWithoutConfidence;

};

}


#endif // KERNELRIDGE_HPP
