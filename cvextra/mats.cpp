#include "mats.hpp"
#include "improc.hpp"
#include "coords.hpp"
#include "LoopRange.hpp"
#include "cvenums.hpp"
#include <opencv2/opencv.hpp>
#include <vector>
#include <algorithm>
#include <initializer_list>

using namespace std;
using namespace cv;


auto cvx::
m(std::initializer_list<std::initializer_list<double>> const& v) -> cv::Mat1d
{
    if (v.size()==0)
    {
        return Mat1d{0,0};
    }

    Mat1d mat(v.size(), (*v.begin()).size());

    int i = 0;
    for (auto const& row : v)
    {
        double* dst = mat[i];
        std::copy(row.begin(), row.end(), dst);
        ++i;
    }

    return mat;
}

auto cvx::mats::
matFromRows(vector<vector<double>> const& v) -> Mat
{
    if (v.empty())
    {
        return Mat1d(0,0);
    }

    Mat1d mat(v.size(), v[0].size());
    cvx::mats::setRows(mat, 0, v);

    return mat;
}

auto cvx::mats::
matFromCols(vector<vector<double>> const& v) -> Mat
{
    return cvx::mats::matFromRows(v).t();
}

auto cvx::mats::
rowsOf(InputArray arr) -> vector<vector<double>>
{
    Mat mat = arr.getMat();
    vector<vector<double>> v(mat.rows);

    for(int row = 0; row < mat.rows; ++row)
    {
        v[row].resize(mat.cols);
        double* matptr = reinterpret_cast<double*>(mat.ptr(row));
        std::copy(matptr, matptr+mat.cols, v[row].begin());
    }

    return v;
}

auto cvx::mats::
checkNaN(Mat1d const& mat, string const& name) -> bool
{
    for (int row=0; row<mat.rows; row++)
    {
        for (int col=0; col<mat.cols; col++)
        {
            if (std::isnan(mat(row,col)))
            {
                cout<< name << " contains NaN at (row="<<row<<", col="<<col<<")"<< endl;

                for (int col2=0; col2<mat.cols; col2++)
                {
                    cout<< col2 << ": "<<mat(row,col2) << endl;
                }

                return true;
            }
        }
    }
    return false;
}

auto cvx::mats::
equals(Mat const& mat1, Mat const& mat2) -> bool
{
    return (&mat1 == &mat2)
            || (mat1.size() == mat2.size()
                && mat1.type() == mat2.type()
                && cv::countNonZero(mat1 != mat2) == 0);
}

/**
 * Converts an Nx1 or a 1xN array into a diagonal NxN matrix
 */
auto cvx::mats::
asDiagonal(InputArray src) -> Mat
{
#if CV_MAJOR_VERSION == 2 && CV_MINOR_VERSION >= 4
    return Mat::diag(src.getMat());
#else
    Mat mat = src.getMat();

    CV_Assert(mat.cols == 1 || mat.rows == 1);
    int len = mat.rows + mat.cols - 1;

    Mat result(len, len, mat.type(), Scalar(0));
    Mat dst = result.diag();

    if (mat.cols == 1)
    {
        mat.copyTo(dst);
    } else
    {
        cv::transpose(mat, dst);
    }

    return result;

#endif

}

void cvx::
waitKey(int keyCode, int milliseconds)
{
    while (cv::waitKey(milliseconds) != keyCode);
}

auto cvx::
reshape(InputArray src, Size newSize) -> Mat
{
    CV_Assert(src.size().area() == newSize.area());
    return src.getMat().reshape(0, newSize.height);
}

auto cvx::
reshapeCols(InputArray src, int targetChannelCount, int targetColumnCount) -> Mat
{
	targetChannelCount = (targetChannelCount == 0 ? src.channels() : targetChannelCount);
	targetColumnCount = (targetColumnCount == 0 ? src.size().width : targetColumnCount);

	CV_Assert((src.total() * src.channels()) % (targetChannelCount * targetColumnCount) == 0);
    int targetRowCount = src.total() * src.channels() / (targetChannelCount * targetColumnCount);

    return src.getMat().reshape(targetChannelCount, targetRowCount);
}

void cvx::
show(Mat image, int delay)
{
    #pragma omp critical
    {
        cv::imshow("default", image);
        cv::waitKey(delay);
    }
}

void cvx::
show(string name, Mat image, int delay)
{
    #pragma omp critical
    {
        cv::imshow(name, image);
        cv::waitKey(delay);
    }
}

void cvx::
vconcat(InputArray src1, InputArray src2, OutputArray dst)
{
    if (src1.empty())
    {
        dst.create(src2.size(), src2.type());
        Mat dstMat = dst.getMat();
        src2.getMat().copyTo(dstMat);
    } else {
        cv::vconcat(src1, src2, dst);
    }
}


void cvx::
hconcat(InputArray src1, InputArray src2, OutputArray dst)
{
    if (src1.empty())
    {
        dst.create(src2.size(), src2.type());
        Mat dstMat = dst.getMat();
        src2.getMat().copyTo(dstMat);
    } else {
        cv::hconcat(src1, src2, dst);
    }
}

auto cvxret::
cumsum(InputArray src, int dim) -> Mat
{
    Mat srcMat = src.getMat();
    Mat result{src.size(), src.type()};

    if (dim == 0)
    {
        Mat row0 = result.row(0);
        srcMat.row(0).copyTo(row0);

        for (int i : cvx::irange(1, srcMat.rows))
        {
            Mat resultRow = result.row(i);
            cv::add(result.row(i-1), srcMat.row(i), resultRow);
        }
    } else if (dim == 1)
    {
        Mat col0 = result.col(0);
        srcMat.col(0).copyTo(col0);

        for (int i : cvx::irange(1, srcMat.cols))
        {
            Mat resultCol = result.col(i);
            cv::add(result.col(i-1), srcMat.col(i), resultCol);
        }
    } else
    {
        throw std::string("Dimension must be 0 or 1");
    }

    return result;
}

auto cvx::
extractRelativeRoi(Mat const& mat, Rectd const& relativeRoi) -> Mat
{
    return mat(cvx::rel2abs(relativeRoi, mat.size()));
}

auto cvxret::
vconcat(InputArray src1, InputArray src2) -> Mat
{
    Mat result;
    cvx::vconcat(src1, src2, result);
    return result;
}

auto cvxret::
hconcat(InputArray src1, InputArray src2) -> Mat
{
    Mat result;
    cvx::hconcat(src1, src2, result);
    return result;
}

auto cv::
clone(cv::Mat const& m) -> std::unique_ptr<cv::Mat>
{
    return std::unique_ptr<cv::Mat>(new cv::Mat(m.clone()));
}

void cvx::
increment(cv::InputArray src1dst, cv::InputArray src2)
{
    cv::add(src1dst, src2, cv::OutputArray(src1dst));
}

void cvx::
medianFilter(
        cv::InputArray src_,
        cv::OutputArray dst_,
        cv::Size kernelSize)
{
    Mat1d src = src_.getMat();
    dst_.create(src.size(), src.type());
    Mat1d dst = dst_.getMat();

    Size kernelWing = (kernelSize-Size{1,1})/2;

    int iMedianInFullWindow = kernelSize.area()/2;
    vector<double> sortTarget(iMedianInFullWindow+1);

    Rect srcRect = cvx::fullRect(src);

    for (Point p : cvx::points(dst))
    {
        Rect rect = cvx::intersect(Rect{p-kernelWing, kernelSize}, srcRect);
        Mat1d srcPart = src(rect);

        int iMedian = rect.area()/2;
        std::partial_sort_copy(srcPart.begin(), srcPart.end(), sortTarget.begin(), sortTarget.begin()+iMedian+1);
        dst(p) = sortTarget[iMedian];
    }
}

void cvx::
variance(
		cv::InputArray src,
		cv::OutputArray dst,
		int dim,
		int dtype,
		cv::InputArray mean_)
{
    Mat mean;
    if (mean_.empty())
    {
        cv::reduce(src, mean, dim, REDUCE_AVG, dtype);
    } else {
        mean = mean_.getMat();
    }

    Mat srcMinusMean =
            src.getMat() - ((dim==0) ?
                    cv::repeat(mean, src.size().height, 1) :
                    cv::repeat(mean, 1, src.size().width));

    return cv::reduce(cvx::sq(srcMinusMean), dst, dim, REDUCE_AVG, dtype);
}



void cvx::
reshapeTo(cv::InputArray _src, cv::OutputArray dst)
{
    assert(_src.channels()*_src.size().area() == dst.channels()*dst.size().area());

    Mat src = _src.getMat();
    if (src.isContinuous())
    {
        src.reshape(dst.channels(), dst.size().height).copyTo(dst);
    } else {
        src.clone().reshape(dst.channels(), dst.size().height).copyTo(dst);
    }
}


void cvx::
backwardDerivative(InputArray src, OutputArray dst, int borderType)
{
    cv::filter2D(src, dst, SRC_TYPE, Matx21d{-1,1}, Point{0,1}, 0, borderType);
}

void cvx::
forwardDerivative(InputArray src, OutputArray dst, int borderType)
{
    cv::filter2D(src, dst, SRC_TYPE, Matx21d{-1,1}, Point{0,0}, 0, borderType);
}

auto cvxret::
backwardDerivative(InputArray src, int borderType) -> Mat
{
    Mat result;
    cvx::backwardDerivative(src, result, borderType);
    return result;
}

auto cvxret::
forwardDerivative(InputArray src, int borderType) -> Mat
{
    Mat result;
    cvx::forwardDerivative(src, result, borderType);
    return result;
}

auto cvxret::
medianFilter(cv::InputArray src, cv::Size kernelSize) -> cv::Mat
{
    Mat result;
    cvx::medianFilter(src, result, kernelSize);
    return result;
}


auto cvxret::
variance(
		cv::InputArray src,
		int dim,
		int dtype,
		cv::InputArray mean_) -> cv::Mat
{
    Mat result;
    cvx::variance(src, result, dim, dtype, mean_);
    return result;
}

auto cvxret::
vconcatAll(std::vector<cv::Mat1d> const& src) -> cv::Mat1d
{
	int nRows = 0;
	for (auto const& elem : src)
	{
		nRows += elem.rows;
	}

	Mat1d result{nRows, src[0].cols};

	int i=0;
	for (auto const& elem : src)
	{
		Mat dst = result.row(i);
		elem.copyTo(dst);
		++i;
	}
	return result;
}

auto cvxret::
hconcatAll(std::vector<cv::Mat1d> const& src) -> cv::Mat1d
{
	Mat1d result{src[0].rows, 0};

	for (auto const& elem : src)
	{
		cvx::hconcat(result, elem, result);
	}
	return result;
}


