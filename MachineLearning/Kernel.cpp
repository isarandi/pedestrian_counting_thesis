#include <MachineLearning/Kernel.hpp>
#include <cvextra/cvenums.hpp>
#include <cvextra/LoopRange.hpp>
#include <cvextra/mats.hpp>
#include <cvextra/math.hpp>
#include <cvextra/cvret.hpp>
#include <armadillo>
#include <cvextra/armadillo.hpp>
#include <opencv2/core/mat.hpp>
#include <cmath>

using namespace cv;
using namespace std;
using namespace crowd;

auto SquaredExponentialKernel::
calcKernelMatrix(Mat1d const& X1, Mat1d const& X2) const -> Mat1d
{
    Mat1d kernelMatrix{X1.rows, X2.rows};

    #pragma omp parallel for
    for (int i=0; i<X1.rows; ++i)
    {
        double const* x1 = reinterpret_cast<double const*>(X1.ptr(i));
        double* ki = reinterpret_cast<double*>(kernelMatrix.ptr(i));

        for (int j=0; j<X2.rows; ++j)
        {
            double const* x2 = reinterpret_cast<double const*>(X2.ptr(j));
            double sumOfSquaredDiffs = 0;

            for (int k = 0; k < X1.cols; ++k)
            {
                sumOfSquaredDiffs += cvx::sq(x1[k]-x2[k]);
            }
            ki[j] = std::exp(-gamma * sumOfSquaredDiffs);
        }
    }

    return kernelMatrix;
}

auto SquaredExponentialKernel::
calcSymmetricKernelMatrix(Mat1d const& X) const -> Mat1d
{
    Mat1d kernelMatrix{X.rows, X.rows};

    #pragma omp parallel for schedule(dynamic)
    for (int i=0; i<X.rows; ++i)
    {
        double const* x1 = reinterpret_cast<double const*>(X.ptr(i));
        double* ki = reinterpret_cast<double*>(kernelMatrix.ptr(i));

        for (int j=0; j<i; ++j)
        {
            double const* x2 = reinterpret_cast<double const*>(X.ptr(j));
            double sumOfSquaredDiffs = 0;

            for (int k = 0; k < X.cols; ++k)
            {
                sumOfSquaredDiffs += cvx::sq(x1[k]-x2[k]);
            }
            ki[j] = std::exp(-gamma * sumOfSquaredDiffs);
            kernelMatrix(j,i) = ki[j];
        }

        ki[i] = 1.0;
    }

    return kernelMatrix;
}

auto SquaredExponentialKernel::
calcDiagonalOfKernelMatrix(Mat1d const& X) const -> Mat1d
{
    return Mat1d::ones(X.rows, 1);
}

auto SquaredExponentialKernel::
calcGradientOfGPMean(
        cv::Mat1d const& K,
        cv::Mat1d const& trainingInputs,
        cv::Mat1d const& testInputs,
        cv::Mat1d const& A
        ) const -> cv::Mat1d
{
    Mat1d K_times_ADiag{K.rows, K.cols};

    Mat1d Acont = (A.isContinuous() ? A : A.clone());
    double const* APtr = reinterpret_cast<double const*>(Acont.data);
    cout << "a" << endl;

    #pragma omp parallel for
    for (int iRow = 0; iRow < testInputs.rows; ++iRow)
    {
        double const* KPtr = reinterpret_cast<double const*>(K.ptr(iRow));
        double* K_times_ADiagPtr = reinterpret_cast<double*>(K_times_ADiag.ptr(iRow));

        for (int iCol = 0; iCol < K.cols; ++iCol)
        {
            K_times_ADiagPtr[iCol] = KPtr[iCol] * APtr[iCol];
        }
    }

    cout << "b" << endl;

    //Mat1d grad = K_times_ADiag * trainingInputs;
    arma::mat armaGradTr = cvret::cv2armaTr(trainingInputs) * cvret::cv2armaTr(K_times_ADiag);
    Mat1d grad = cvret::arma2cvTr(armaGradTr);

    //Mat1d Ka = K * Acont;
    arma::mat armaKaTr = cvret::cv2armaTr(Acont) * cvret::cv2armaTr(K);
    Mat1d Ka = cvret::arma2cvTr(armaKaTr);

    double twoGamma = 2*gamma;
    cout << "c" << endl;

    #pragma omp parallel for
    for (int iRow = 0; iRow < testInputs.rows; ++iRow)
    {
        double factor = Ka(iRow);
        double* gradPtr = reinterpret_cast<double*>(grad.ptr(iRow));
        double const* testInputsPtr = reinterpret_cast<double const*>(testInputs.ptr(iRow));

        for (int iCol = 0; iCol < testInputs.cols; ++iCol)
        {
            gradPtr[iCol] -= factor * testInputsPtr[iCol];
            gradPtr[iCol] *= twoGamma;
        }
    }

    cout << "d" << endl;

    return grad.clone();
}
