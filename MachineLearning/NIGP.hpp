#ifndef MACHINELEARNING_NIGP_HPP_
#define MACHINELEARNING_NIGP_HPP_

#include <MachineLearning/Kernel.hpp>
#include <MachineLearning/LearningSet.hpp>
#include <MachineLearning/Regression.hpp>
#include <MachineLearning/KernelRidge.hpp>
#include <opencv2/core/core.hpp>
#include <stdx/cloning.hpp>
//#include <Eigen/Core>
//#include <Eigen/Cholesky>
#include <armadillo>
#include <memory>
#include <string>
#include <vector>

namespace crowd {

class NIGP : public RegressionWithConfidence
{
public:
    NIGP(
            KernelRidgeParams const& params,
            double rbfGamma,
			double inputNoise,
			bool predictWithoutConfidence = false,
			int nIterations = 1);

    // Regression interface
    virtual void train(LearningSet const& ls);
    virtual auto predict(cv::Mat1d const& X) const -> cv::Mat1d;
    virtual auto predictWithConfidence(cv::Mat1d const& X) const -> PredictionWithConfidence;

    virtual auto getJacobian(cv::Mat1d const& X) const -> cv::Mat1d;

    virtual auto getDescription() const -> std::string;

    CVX_CLONE_IN_DERIVED(NIGP)
    CVX_CONFIG_DERIVED(NIGP)

private:
    void solveForA(
            cv::Mat1d const& regularizedK,
            cv::Mat1d const& trainingOutputs,
			int iOutput);

    KernelRidgeParams params;
    stdx::cloned_unique_ptr<Kernel> kernel;

//    typedef decltype(Eigen::MatrixXd().llt()) EigenLLT;
//    std::vector<stdx::copy_constructed_unique_ptr<EigenLLT>> llts; // = Cholesky((K+.)^-1)

    std::vector<arma::mat> Ls;

    cv::Mat1d A; // = (K+.)^-1 * Y
    cv::Mat1d trainingInputs;

    cv::Mat1d inputNoiseVariances;
    double inputNoiseVarianceConstant;

    bool predictWithoutConfidence;

    int nIterations;

};

}

#endif /* MACHINELEARNING_NIGP_HPP_ */
