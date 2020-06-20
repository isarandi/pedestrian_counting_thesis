#ifndef MATRIXUTILS_HPP
#define MATRIXUTILS_HPP

#include "core.hpp"
#include <opencv2/opencv.hpp>
#include <vector>
#include <memory>
#include <cstdlib>

namespace cvx {

void variance(cv::InputArray src, cv::OutputArray dst, int dim, int dtype=-1, cv::InputArray mean=cv::noArray());

void vconcat(cv::InputArray src1, cv::InputArray src2, cv::OutputArray dst);
void hconcat(cv::InputArray src1, cv::InputArray src2, cv::OutputArray dst);

auto m(std::initializer_list<std::initializer_list<double>> const& v) -> cv::Mat1d;

auto reshape(cv::InputArray src, cv::Size s) -> cv::Mat;
auto reshapeCols(cv::InputArray src, int targetChannelCount, int targetColumnCount) -> cv::Mat;

void reshapeTo(cv::InputArray src, cv::OutputArray dst);

template <typename T>
auto FrobeniusSq(T&& m) -> decltype(m.dot(m))
{
	return m.dot(m);
}

void backwardDerivative(cv::InputArray src, cv::OutputArray dst, int borderType=cv::BORDER_CONSTANT);
void forwardDerivative(cv::InputArray src, cv::OutputArray dst, int borderType=cv::BORDER_CONSTANT);

void increment(cv::InputArray src1dst, cv::InputArray src2);

void show(cv::Mat image, int delay = 0);
void show(std::string name, cv::Mat image, int delay = 0);

void waitKey(int keyCode, int milliseconds = 0);

void medianFilter(cv::InputArray src, cv::OutputArray dst, cv::Size kernelSize);

auto extractRelativeRoi(cv::Mat const& mat, cvx::Rectd const& relativeRoi) -> cv::Mat;

    namespace mats
    {

        auto rowsOf(cv::InputArray arr) -> std::vector<std::vector<double>>;

        void setRow(cv::Mat1d& mat, int row, int offset, std::vector<double> const& v);
        void setRow(cv::Mat1d& mat, int row, std::vector<double> const& v);
        void setRows(cv::Mat1d& mat, int firstRow, std::vector<std::vector<double>> const& v);

        auto matFromRows(std::vector<std::vector<double>> const& v) -> cv::Mat;
        auto matFromCols(std::vector<std::vector<double>> const& v) -> cv::Mat;
        auto asDiagonal(cv::InputArray src) -> cv::Mat;

        auto checkNaN(cv::Mat1d const& mat, std::string const& name) -> bool;
        auto equals(cv::Mat const& mat1, cv::Mat const& mat2) -> bool;
    }

    /**
     * Mat_::operator() was changed in OpenCV 2.4. This function allows compatibility
     * by defining the new version's semantic as a global function.
     */
    template <typename _Tp> inline
    auto at(cv::Mat_<_Tp>& mat, int i0) -> _Tp&
    {
        #if CV_MAJOR_VERSION == 2 && CV_MINOR_VERSION >= 4
        return mat(i0);

        #else
        CV_DbgAssert( mat.dims <= 2 && mat.data &&
                     (unsigned)i0 < (unsigned)(mat.size.p[0] * mat.size.p[1]) &&
                     mat.elemSize() == CV_ELEM_SIZE(cv::DataType<_Tp>::type) );
        if( mat.isContinuous() || mat.size.p[0] == 1 )
            return ((_Tp*)mat.data)[i0];
        if( mat.size.p[1] == 1 )
            return *(_Tp*)(mat.data + mat.step.p[0] * i0);
        int i = i0 / mat.cols, j = i0 - i * mat.cols;
        return ((_Tp*)(mat.data + mat.step.p[0] * i))[j];

        #endif
    }

    /**
     * Mat_::operator() was changed in OpenCV 2.4. This function allows compatibility
     * by defining the new version's semantic as a global function.
     */
    template <typename _Tp> inline
    auto at(const cv::Mat_<_Tp>& mat, int i0) -> _Tp const&
    {
        #if CV_MAJOR_VERSION == 2 && CV_MINOR_VERSION >= 4
        return mat(i0);
        #else
        CV_DbgAssert( mat.dims <= 2 && mat.data &&
                     (unsigned)i0 < (unsigned)(mat.size.p[0] * mat.size.p[1]) &&
                     mat.elemSize() == CV_ELEM_SIZE(cv::DataType<_Tp>::type) );
        if( mat.isContinuous() || mat.size.p[0] == 1 )
            return ((const _Tp*)mat.data)[i0];
        if( mat.size.p[1] == 1 )
            return *(const _Tp*)(mat.data + mat.step.p[0] * i0);
        int i = i0 / mat.cols, j = i0 - i * mat.cols;
        return ((const _Tp*)(mat.data + mat.step.p[0] * i))[j];
        #endif
    }




}

namespace cvxret {
    auto vconcat(cv::InputArray src1, cv::InputArray src2) -> cv::Mat;
    auto hconcat(cv::InputArray src1, cv::InputArray src2) -> cv::Mat;

    auto vconcatAll(std::vector<cv::Mat1d> const& src) -> cv::Mat1d;
    auto hconcatAll(std::vector<cv::Mat1d> const& src) -> cv::Mat1d;

    auto cumsum(cv::InputArray src, int dim = 0) -> cv::Mat;
    auto variance(cv::InputArray src, int dim, int dtype=-1, cv::InputArray mean=cv::noArray()) -> cv::Mat;

    auto backwardDerivative(
            cv::InputArray src,
            int borderType=cv::BORDER_CONSTANT
            ) -> cv::Mat;

    auto forwardDerivative(
            cv::InputArray src,
            int borderType=cv::BORDER_CONSTANT
            ) -> cv::Mat;

    auto medianFilter(
    		cv::InputArray src,
			cv::Size kernelSize
			) -> cv::Mat;
}

inline
void cvx::mats::
setRow(cv::Mat1d& mat, int row, int offset, std::vector<double> const& v)
{
    double* dst = mat[row] + offset;
    std::copy(v.begin(), v.end(), dst);
}

inline
void cvx::mats::
setRow(cv::Mat1d& mat, int row, std::vector<double> const& v)
{
    setRow(mat, row, 0, v);
}

inline
void cvx::mats::
setRows(cv::Mat1d& mat, int firstRow, std::vector<std::vector<double>> const& v)
{
    int n = v.size();
    for (int row = 0; row < n; ++row)
    {
        setRow(mat, row + firstRow, v[row]);
    }
}

namespace cv {

    /**
     * @brief For use with the cloning functionality in stdx
     */
    auto clone(cv::Mat const& m) -> std::unique_ptr<cv::Mat>;

//    auto move_(cv::Mat const& m) -> std::unique_ptr<cv::Mat>;

    /**
     * @brief For use with the cloning functionality in stdx
     */
    template <typename _Tp>
    auto clone(cv::Mat_<_Tp> const& m) -> std::unique_ptr<cv::Mat_<_Tp>>
    {
        return std::unique_ptr<cv::Mat_<_Tp>>(new cv::Mat_<_Tp>(m.clone()));
    }

//    template <typename _Tp>
//    auto move_(cv::Mat_<_Tp> const& m) -> std::unique_ptr<cv::Mat_<_Tp>>
//    {
//        return std::unique_ptr<cv::Mat_<_Tp>>(new cv::Mat_<_Tp>(m));
//    }
}

#endif // MATRIXUTILS_HPP
