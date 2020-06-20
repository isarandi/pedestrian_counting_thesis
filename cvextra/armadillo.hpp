#ifndef CVEXTRA_ARMADILLO_H_
#define CVEXTRA_ARMADILLO_H_

#include <armadillo>
#include <opencv2/core/core.hpp>

namespace cvret {

inline
auto arma2cvTr(arma::mat const& armamat) -> cv::Mat1d
{
    return cv::Mat1d(armamat.n_cols, armamat.n_rows, const_cast<double*>(armamat.memptr()));
}

inline
auto cv2armaTr(cv::Mat1d const& cvmat) -> arma::mat
{
    return arma::mat(reinterpret_cast<double*>(cvmat.data), cvmat.cols, cvmat.rows, false);
}

inline
auto arma2cv(arma::mat const& armamat) -> cv::Mat1d
{
    if ((armamat.n_rows == 1 || armamat.n_cols==1))
    {
        return cv::Mat1d(armamat.n_rows, armamat.n_cols, const_cast<double*>(armamat.memptr()));
    }

    return cv::Mat1d(armamat.n_cols, armamat.n_rows, const_cast<double*>(armamat.memptr())).t();
}

inline
auto cv2arma(cv::Mat1d const& cvmat) -> arma::mat
{
    if ((cvmat.rows == 1 || cvmat.cols==1) && cvmat.isContinuous())
    {
        return arma::mat(reinterpret_cast<double*>(cvmat.data), cvmat.rows, cvmat.cols, false);
    }

    arma::mat result(cvmat.rows, cvmat.cols);
    cv::transpose(cvmat, cvret::arma2cvTr(result));
    return result;
}





}



#endif /* CVEXTRA_ARMADILLO_H_ */
