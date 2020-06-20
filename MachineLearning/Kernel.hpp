#ifndef MACHINELEARNING_KERNEL_HPP_
#define MACHINELEARNING_KERNEL_HPP_

#include <opencv2/core/core.hpp>
#include <stdx/cloning.hpp>



namespace crowd {

class Kernel
{
public:
    virtual auto
    calcKernelMatrix(cv::Mat1d const& X1, cv::Mat1d const& X2) const -> cv::Mat1d = 0;

    virtual auto
    calcSymmetricKernelMatrix(cv::Mat1d const& X) const -> cv::Mat1d = 0;

    virtual auto
    calcDiagonalOfKernelMatrix(cv::Mat1d const& X) const -> cv::Mat1d = 0;

    virtual auto
    calcGradientOfGPMean(
            cv::Mat1d const& K,
            cv::Mat1d const& trainingInputs,
            cv::Mat1d const& testInputs,
            cv::Mat1d const& A
            ) const -> cv::Mat1d = 0;

    CVX_CLONE_IN_BASE(Kernel)
};

class SquaredExponentialKernel : public Kernel
{
public:
    explicit SquaredExponentialKernel(double rbfGamma)
        : gamma(rbfGamma){}

    virtual auto
    calcKernelMatrix(cv::Mat1d const& X1, cv::Mat1d const& X2) const -> cv::Mat1d;

    virtual auto
    calcSymmetricKernelMatrix(cv::Mat1d const& X) const -> cv::Mat1d;

    virtual auto
    calcDiagonalOfKernelMatrix(cv::Mat1d const& X) const -> cv::Mat1d;

    virtual auto
    calcGradientOfGPMean(
            cv::Mat1d const& K,
            cv::Mat1d const& trainingInputs,
            cv::Mat1d const& testInputs,
            cv::Mat1d const& A
            ) const -> cv::Mat1d;

    CVX_CLONE_IN_DERIVED(SquaredExponentialKernel)
private:
    double gamma;
};

} /* namespace crowd */

#endif /* MACHINELEARNING_KERNEL_HPP_ */
