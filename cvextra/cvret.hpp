#ifndef CVRET_HPP
#define CVRET_HPP

#include "cvenums.hpp"
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <vector>

namespace cvret
{

cv::Mat convertType(cv::InputArray src, int dstType);
cv::Mat variance(cv::InputArray src, int dim, int dtype=-1, cv::InputArray mean=cv::noArray());
cv::Mat magnitude(cv::InputArray xy);

std::vector<double> HuMoments(const cv::Moments& m);
cv::Mat rectangle(cv::InputArray img, cv::Rect rec, const cv::Scalar& color, int thickness=1, int lineType=8, int shift=0);
cv::Mat line(cv::InputArray img, cv::Point pt1, cv::Point pt2, const cv::Scalar& color, int thickness=1, int lineType=8, int shift=0);
cv::Mat setTo(cv::InputArray img, cv::Scalar value, cv::InputArray mask);





inline std::vector<cv::Mat> split(cv::Mat const& src);

cv::Mat copyMakeBorder(cv::InputArray src, int top, int bottom, int left, int right, int borderType, const cv::Scalar& value=cv::Scalar());
cv::Mat medianBlur(cv::InputArray src, int ksize);
cv::Mat GaussianBlur(cv::InputArray src, cv::Size ksize, double sigma1, double sigma2=0, int borderType=cv::BORDER_DEFAULT);
cv::Mat bilateralFilter(cv::InputArray src, int d, double sigmaColor, double sigmaSpace, int borderType=cv::BORDER_DEFAULT);
cv::Mat boxFilter(cv::InputArray src, int ddepth, cv::Size ksize, cv::Point anchor=cv::Point(-1,-1), bool normalize=true, int borderType=cv::BORDER_DEFAULT);
cv::Mat blur(cv::InputArray src, cv::Size ksize, cv::Point anchor=cv::Point(-1,-1), int borderType=cv::BORDER_DEFAULT);
cv::Mat filter2D(cv::InputArray src, int ddepth, cv::InputArray kernel, cv::Point anchor=cv::Point(-1,-1), double delta=0, int borderType=cv::BORDER_DEFAULT);
cv::Mat sepFilter2D(cv::InputArray src, int ddepth, cv::InputArray kernelX, cv::InputArray kernelY, cv::Point anchor=cv::Point(-1,-1), double delta=0, int borderType=cv::BORDER_DEFAULT);
cv::Mat Sobel(cv::InputArray src, int ddepth, int dx, int dy, int ksize=3, double scale=1, double delta=0, int borderType=cv::BORDER_DEFAULT);
cv::Mat Scharr(cv::InputArray src, int ddepth, int dx, int dy, double scale=1, double delta=0, int borderType=cv::BORDER_DEFAULT);
cv::Mat Laplacian(cv::InputArray src, int ddepth, int ksize=1, double scale=1, double delta=0, int borderType=cv::BORDER_DEFAULT);
cv::Mat Canny(cv::InputArray image, double threshold1, double threshold2, int apertureSize=3, bool L2gradient=false);
cv::Mat cornerMinEigenVal(cv::InputArray src, int blockSize, int ksize=3, int borderType=cv::BORDER_DEFAULT);
cv::Mat cornerHarris(cv::InputArray src, int blockSize, int ksize, double k, int borderType=cv::BORDER_DEFAULT);
cv::Mat cornerEigenValsAndVecs(cv::InputArray src, int blockSize, int ksize, int borderType=cv::BORDER_DEFAULT);
cv::Mat preCornerDetect(cv::InputArray src, int ksize, int borderType=cv::BORDER_DEFAULT);
cv::Mat cornerSubPix(cv::InputArray image, cv::Size winSize, cv::Size zeroZone, cv::TermCriteria criteria);
cv::Mat goodFeaturesToTrack(cv::InputArray image, int maxCorners, double qualityLevel, double minDistance, cv::InputArray mask=cv::noArray(), int blockSize=3, bool useHarrisDetector=false, double k=0.04);
cv::Mat HoughLines(cv::InputArray image, double rho, double theta, int threshold, double srn=0, double stn=0);
cv::Mat HoughLinesP(cv::InputArray image, double rho, double theta, int threshold, double minLineLength=0, double maxLineGap=0);
cv::Mat HoughCircles(cv::InputArray image, int method, double dp, double minDist, double param1=100, double param2=100, int minRadius=0, int maxRadius=0);
cv::Mat erode(cv::InputArray src, cv::InputArray kernel, cv::Point anchor=cv::Point(-1,-1), int iterations=1, int borderType=cv::BORDER_CONSTANT, const cv::Scalar& borderValue=cv::morphologyDefaultBorderValue());
cv::Mat dilate(cv::InputArray src, cv::InputArray kernel, cv::Point anchor=cv::Point(-1,-1), int iterations=1, int borderType=cv::BORDER_CONSTANT, const cv::Scalar& borderValue=cv::morphologyDefaultBorderValue());
cv::Mat morphologyEx(cv::InputArray src, int op, cv::InputArray kernel, cv::Point anchor=cv::Point(-1,-1), int iterations=1, int borderType=cv::BORDER_CONSTANT, const cv::Scalar& borderValue=cv::morphologyDefaultBorderValue());
cv::Mat resize(cv::InputArray src, cv::Size dsize, double fx=0, double fy=0, int interpolation=cv::INTER_LINEAR);
cv::Mat warpAffine(cv::InputArray src, cv::InputArray M, cv::Size dsize, int flags=cv::INTER_LINEAR, int borderMode=cv::BORDER_CONSTANT, const cv::Scalar& borderValue=cv::Scalar());
cv::Mat warpPerspective(cv::InputArray src, cv::InputArray M, cv::Size dsize, int flags=cv::INTER_LINEAR, int borderMode=cv::BORDER_CONSTANT, const cv::Scalar& borderValue=cv::Scalar());
cv::Mat remap(cv::InputArray src, cv::InputArray map1, cv::InputArray map2, int interpolation, int borderMode=cv::BORDER_CONSTANT, const cv::Scalar& borderValue=cv::Scalar());
cv::Mat invertAffineTransform(cv::InputArray M);
cv::Mat getRectSubPix(cv::InputArray image, cv::Size patchSize, cv::Point2f center, int patchType=- 1);
cv::Mat integral(cv::InputArray src, int sdepth=- 1);
cv::Mat accumulate(cv::InputArray src, cv::InputArray mask=cv::noArray());
cv::Mat accumulateSquare(cv::InputArray src, cv::InputArray mask=cv::noArray());
cv::Mat accumulateProduct(cv::InputArray src1, cv::InputArray src2, cv::InputArray mask=cv::noArray());
cv::Mat accumulateWeighted(cv::InputArray src, double alpha, cv::InputArray mask=cv::noArray());
cv::Mat threshold(cv::InputArray src, double thresh, double maxval, int type);
cv::Mat adaptiveThreshold(cv::InputArray src, double maxValue, int adaptiveMethod, int thresholdType, int blockSize, double C);
cv::Mat pyrDown(cv::InputArray src, const cv::Size& dstsize=cv::Size(), int borderType=cv::BORDER_DEFAULT);
cv::Mat pyrUp(cv::InputArray src, const cv::Size& dstsize=cv::Size(), int borderType=cv::BORDER_DEFAULT);
cv::Mat undistort(cv::InputArray src, cv::InputArray cameraMatrix, cv::InputArray distCoeffs, cv::InputArray newCameraMatrix=cv::noArray());
cv::Mat calcHist(cv::InputArrayOfArrays images, const std::vector<int>& channels, cv::InputArray mask, const std::vector<int>& histSize, const std::vector<float>& ranges, bool accumulate=false);
cv::Mat calcBackProject(cv::InputArrayOfArrays images, const std::vector<int>& channels, cv::InputArray hist, const std::vector<float>& ranges, double scale);
cv::Mat equalizeHist(cv::InputArray src);
cv::Mat watershed(cv::InputArray image);
cv::Mat pyrMeanShiftFiltering(cv::InputArray src, double sp, double sr, int maxLevel=1, cv::TermCriteria termcrit=cv::TermCriteria(cv::TermCriteria::MAX_ITER+cv::TermCriteria::EPS,5,1));
cv::Mat cvtColor(cv::InputArray src, int code, int dstCn=0);
cv::Mat matchTemplate(cv::InputArray image, cv::InputArray templ, int method);
cv::Mat drawContours(cv::InputArrayOfArrays contours, int contourIdx, const cv::Scalar& color, int thickness=1, int lineType=8, cv::InputArray hierarchy=cv::noArray(), int maxLevel=INT_MAX, cv::Point offset=cv::Point());
cv::Mat approxPolyDP(cv::InputArray curve, double epsilon, bool closed);
cv::Mat convexHull(cv::InputArray points, bool clockwise=false, bool returnPoints=true);
cv::Mat fitLine(cv::InputArray points, int distType, double param, double reps, double aeps);
cv::Mat add(cv::InputArray src1, cv::InputArray src2, cv::InputArray mask=cv::noArray(), int dtype=- 1);
cv::Mat subtract(cv::InputArray src1, cv::InputArray src2, cv::InputArray mask=cv::noArray(), int dtype=- 1);
cv::Mat multiply(cv::InputArray src1, cv::InputArray src2, double scale=1, int dtype=- 1);
cv::Mat divide(cv::InputArray src1, cv::InputArray src2, double scale=1, int dtype=- 1);
cv::Mat divide(double scale, cv::InputArray src2, int dtype=- 1);
cv::Mat scaleAdd(cv::InputArray src1, double alpha, cv::InputArray src2);
cv::Mat addWeighted(cv::InputArray src1, double alpha, cv::InputArray src2, double beta, double gamma, int dtype=- 1);
cv::Mat convertScaleAbs(cv::InputArray src, double alpha=1, double beta=0);
cv::Mat normalize(cv::InputArray src, double alpha=1, double beta=0, int norm_type=cv::NORM_L2, int dtype=-1, cv::InputArray mask=cv::noArray());
cv::Mat reduce(cv::InputArray src, int dim, int rtype, int dtype=-1);
cv::Mat merge(const std::vector<cv::Mat>& mv);
cv::Mat extractChannel(cv::InputArray src, int coi);
cv::Mat insertChannel(cv::InputArray src, int coi);
cv::Mat flip(cv::InputArray src, int flipCode);
cv::Mat repeat(cv::InputArray src, int ny, int nx);
cv::Mat hconcat(cv::InputArray src);
cv::Mat vconcat(cv::InputArray src);
cv::Mat hconcat(cv::InputArray src1, cv::InputArray src2);
cv::Mat vconcat(cv::InputArray src1, cv::InputArray src2);
cv::Mat bitwise_and(cv::InputArray src1, cv::InputArray src2, cv::InputArray mask=cv::noArray());
cv::Mat bitwise_or(cv::InputArray src1, cv::InputArray src2, cv::InputArray mask=cv::noArray());
cv::Mat bitwise_xor(cv::InputArray src1, cv::InputArray src2, cv::InputArray mask=cv::noArray());
cv::Mat bitwise_not(cv::InputArray src, cv::InputArray mask=cv::noArray());
cv::Mat absdiff(cv::InputArray src1, cv::InputArray src2);
cv::Mat inRange(cv::InputArray src, cv::InputArray lowerb, cv::InputArray upperb);
cv::Mat compare(cv::InputArray src1, cv::InputArray src2, int cmpop);
cv::Mat sqrt(cv::InputArray src);
cv::Mat pow(cv::InputArray src, double power);
cv::Mat exp(cv::InputArray src);
cv::Mat log(cv::InputArray src);
cv::Mat phase(cv::InputArray x, cv::InputArray y, bool angleInDegrees=false);
cv::Mat magnitude(cv::InputArray x, cv::InputArray y);
cv::Mat gemm(cv::InputArray src1, cv::InputArray src2, double alpha, cv::InputArray src3, double gamma, int flags=0);
cv::Mat mulTransposed(cv::InputArray src, bool aTa, cv::InputArray delta=cv::noArray(), double scale=1, int dtype=- 1);
cv::Mat transpose(cv::InputArray src);
cv::Mat transform(cv::InputArray src, cv::InputArray m);
cv::Mat perspectiveTransform(cv::InputArray src, cv::InputArray m);
cv::Mat completeSymm(bool lowerToUpper=false);
cv::Mat setIdentity(const cv::Scalar& s=cv::Scalar(1));
cv::Mat invert(cv::InputArray src, int flags=cv::DECOMP_LU);
cv::Mat solve(cv::InputArray src1, cv::InputArray src2, int flags=cv::DECOMP_LU);
cv::Mat sort(cv::InputArray src, int flags);
cv::Mat sortIdx(cv::InputArray src, int flags);
cv::Mat solveCubic(cv::InputArray coeffs);
cv::Mat solvePoly(cv::InputArray coeffs, int maxIters=300);
cv::Mat PCAProject(cv::InputArray data, cv::InputArray mean, cv::InputArray eigenvectors);
cv::Mat PCABackProject(cv::InputArray data, cv::InputArray mean, cv::InputArray eigenvectors);
cv::Mat SVBackSubst(cv::InputArray w, cv::InputArray u, cv::InputArray vt, cv::InputArray rhs);
cv::Mat dft(cv::InputArray src, int flags=0, int nonzeroRows=0);
cv::Mat idft(cv::InputArray src, int flags=0, int nonzeroRows=0);
cv::Mat dct(cv::InputArray src, int flags=0);
cv::Mat idct(cv::InputArray src, int flags=0);
cv::Mat mulSpectrums(cv::InputArray a, cv::InputArray b, int flags, bool conjB=false);
cv::Mat fillConvexPoly(cv::InputArray points, const cv::Scalar& color, int lineType=8, int shift=0);
cv::Mat fillPoly(cv::InputArrayOfArrays pts, const cv::Scalar& color, int lineType=8, int shift=0, cv::Point offset=cv::Point());
cv::Mat polylines(cv::InputArrayOfArrays pts, bool isClosed, const cv::Scalar& color, int thickness=1, int lineType=8, int shift=0);


template<typename _Tp, int D> inline
std::vector<cv::Mat_<_Tp>> split(cv::Mat_<cv::Vec<_Tp,D>> const& src)
{
    std::vector<cv::Mat_<_Tp>> result;
    cv::split(src, result);
    return result;
}

}

inline cv::Mat cvret::variance(cv::InputArray src, int dim, int dtype, cv::InputArray mean_)
{
    cv::Mat mean;
    if (mean_.empty())
    {
        cv::reduce(src, mean, dim, cv::REDUCE_AVG, dtype);
    } else {
        mean = mean_.getMat();
    }

    cv::Mat srcMinusMean =
            src.getMat() - ((dim==0) ?
                    cv::repeat(mean, src.size().height, 1) :
                    cv::repeat(mean, 1, src.size().width));

    return cvret::reduce(srcMinusMean.mul(srcMinusMean), dim, cv::REDUCE_AVG, dtype);
}


inline cv::Mat cvret::convertType(cv::InputArray src, int dstType)
{
    cv::Mat result;
    src.getMat().convertTo(result, dstType);
    return result;
}


inline std::vector<double> cvret::HuMoments(const cv::Moments& m)
{
    std::vector<double> result;
    cv::HuMoments(m, result);
    return result;
}

inline cv::Mat cvret::rectangle(cv::InputArray img, cv::Rect rec, cv::Scalar const& color, int thickness, int lineType, int shift)
{
    cv::Mat copy = img.getMat().clone();
    cv::rectangle(copy, rec, color, thickness, lineType, shift);
    return copy;
}

inline cv::Mat cvret::line(cv::InputArray img, cv::Point pt1, cv::Point pt2, const cv::Scalar &color, int thickness, int lineType, int shift)
{
    cv::Mat copy = img.getMat().clone();
    cv::line(copy, pt1, pt2, color, thickness, lineType, shift);
    return copy;
}


inline std::vector<cv::Mat> cvret::split(cv::Mat const& src)
{
    std::vector<cv::Mat> result;
    cv::split(src, result);
    return result;
}


//------------- below autogenerated

inline cv::Mat cvret::copyMakeBorder(cv::InputArray src, int top, int bottom, int left, int right, int borderType, const cv::Scalar& value)
{
    cv::Mat result;
    cv::copyMakeBorder(src, result, top, bottom, left, right, borderType, value);
    return result;
}

inline cv::Mat cvret::medianBlur(cv::InputArray src, int ksize)
{
    cv::Mat result;
    cv::medianBlur(src, result, ksize);
    return result;
}

inline cv::Mat cvret::GaussianBlur(cv::InputArray src, cv::Size ksize, double sigma1, double sigma2, int borderType)
{
    cv::Mat result;
    cv::GaussianBlur(src, result, ksize, sigma1, sigma2, borderType);
    return result;
}



inline cv::Mat cvret::bilateralFilter(cv::InputArray src, int d, double sigmaColor, double sigmaSpace, int borderType)
{
    cv::Mat result;
    cv::bilateralFilter(src, result, d, sigmaColor, sigmaSpace, borderType);
    return result;
}

inline cv::Mat cvret::boxFilter(cv::InputArray src, int ddepth, cv::Size ksize, cv::Point anchor, bool normalize, int borderType)
{
    cv::Mat result;
    cv::boxFilter(src, result, ddepth, ksize, anchor, normalize, borderType);
    return result;
}

inline cv::Mat cvret::blur(cv::InputArray src, cv::Size ksize, cv::Point anchor, int borderType)
{
    cv::Mat result;
    cv::blur(src, result, ksize, anchor, borderType);
    return result;
}

inline cv::Mat cvret::filter2D(cv::InputArray src, int ddepth, cv::InputArray kernel, cv::Point anchor, double delta, int borderType)
{
    cv::Mat result;
    cv::filter2D(src, result, ddepth, kernel, anchor, delta, borderType);
    return result;
}

inline cv::Mat cvret::sepFilter2D(cv::InputArray src, int ddepth, cv::InputArray kernelX, cv::InputArray kernelY, cv::Point anchor, double delta, int borderType)
{
    cv::Mat result;
    cv::sepFilter2D(src, result, ddepth, kernelX, kernelY, anchor, delta, borderType);
    return result;
}

inline cv::Mat cvret::Sobel(cv::InputArray src, int ddepth, int dx, int dy, int ksize, double scale, double delta, int borderType)
{
    cv::Mat result;
    cv::Sobel(src, result, ddepth, dx, dy, ksize, scale, delta, borderType);
    return result;
}

inline cv::Mat cvret::Scharr(cv::InputArray src, int ddepth, int dx, int dy, double scale, double delta, int borderType)
{
    cv::Mat result;
    cv::Scharr(src, result, ddepth, dx, dy, scale, delta, borderType);
    return result;
}

inline cv::Mat cvret::Laplacian(cv::InputArray src, int ddepth, int ksize, double scale, double delta, int borderType)
{
    cv::Mat result;
    cv::Laplacian(src, result, ddepth, ksize, scale, delta, borderType);
    return result;
}

inline cv::Mat cvret::Canny(cv::InputArray image, double threshold1, double threshold2, int apertureSize, bool L2gradient)
{
    cv::Mat result;
    cv::Canny(image, result, threshold1, threshold2, apertureSize, L2gradient);
    return result;
}

inline cv::Mat cvret::cornerMinEigenVal(cv::InputArray src, int blockSize, int ksize, int borderType)
{
    cv::Mat result;
    cv::cornerMinEigenVal(src, result, blockSize, ksize, borderType);
    return result;
}

inline cv::Mat cvret::cornerHarris(cv::InputArray src, int blockSize, int ksize, double k, int borderType)
{
    cv::Mat result;
    cv::cornerHarris(src, result, blockSize, ksize, k, borderType);
    return result;
}

inline cv::Mat cvret::cornerEigenValsAndVecs(cv::InputArray src, int blockSize, int ksize, int borderType)
{
    cv::Mat result;
    cv::cornerEigenValsAndVecs(src, result, blockSize, ksize, borderType);
    return result;
}

inline cv::Mat cvret::preCornerDetect(cv::InputArray src, int ksize, int borderType)
{
    cv::Mat result;
    cv::preCornerDetect(src, result, ksize, borderType);
    return result;
}

inline cv::Mat cvret::cornerSubPix(cv::InputArray image, cv::Size winSize, cv::Size zeroZone, cv::TermCriteria criteria)
{
    cv::Mat result;
    cv::cornerSubPix(image, result, winSize, zeroZone, criteria);
    return result;
}

inline cv::Mat cvret::goodFeaturesToTrack(cv::InputArray image, int maxCorners, double qualityLevel, double minDistance, cv::InputArray mask, int blockSize, bool useHarrisDetector, double k)
{
    cv::Mat result;
    cv::goodFeaturesToTrack(image, result, maxCorners, qualityLevel, minDistance, mask, blockSize, useHarrisDetector, k);
    return result;
}

inline cv::Mat cvret::HoughLines(cv::InputArray image, double rho, double theta, int threshold, double srn, double stn)
{
    cv::Mat result;
    cv::HoughLines(image, result, rho, theta, threshold, srn, stn);
    return result;
}

inline cv::Mat cvret::HoughLinesP(cv::InputArray image, double rho, double theta, int threshold, double minLineLength, double maxLineGap)
{
    cv::Mat result;
    cv::HoughLinesP(image, result, rho, theta, threshold, minLineLength, maxLineGap);
    return result;
}

inline cv::Mat cvret::HoughCircles(cv::InputArray image, int method, double dp, double minDist, double param1, double param2, int minRadius, int maxRadius)
{
    cv::Mat result;
    cv::HoughCircles(image, result, method, dp, minDist, param1, param2, minRadius, maxRadius);
    return result;
}

inline cv::Mat cvret::erode(cv::InputArray src, cv::InputArray kernel, cv::Point anchor, int iterations, int borderType, const cv::Scalar& borderValue)
{
    cv::Mat result;
    cv::erode(src, result, kernel, anchor, iterations, borderType, borderValue);
    return result;
}

inline cv::Mat cvret::dilate(cv::InputArray src, cv::InputArray kernel, cv::Point anchor, int iterations, int borderType, const cv::Scalar& borderValue)
{
    cv::Mat result;
    cv::dilate(src, result, kernel, anchor, iterations, borderType, borderValue);
    return result;
}

inline cv::Mat cvret::morphologyEx(cv::InputArray src, int op, cv::InputArray kernel, cv::Point anchor, int iterations, int borderType, const cv::Scalar& borderValue)
{
    cv::Mat result;
    cv::morphologyEx(src, result, op, kernel, anchor, iterations, borderType, borderValue);
    return result;
}

inline cv::Mat cvret::resize(cv::InputArray src, cv::Size dsize, double fx, double fy, int interpolation)
{
    cv::Mat result;
    cv::resize(src, result, dsize, fx, fy, interpolation);
    return result;
}

inline cv::Mat cvret::warpAffine(cv::InputArray src, cv::InputArray M, cv::Size dsize, int flags, int borderMode, const cv::Scalar& borderValue)
{
    cv::Mat result;
    cv::warpAffine(src, result, M, dsize, flags, borderMode, borderValue);
    return result;
}

inline cv::Mat cvret::warpPerspective(cv::InputArray src, cv::InputArray M, cv::Size dsize, int flags, int borderMode, const cv::Scalar& borderValue)
{
    cv::Mat result;
    cv::warpPerspective(src, result, M, dsize, flags, borderMode, borderValue);
    return result;
}

inline cv::Mat cvret::remap(cv::InputArray src, cv::InputArray map1, cv::InputArray map2, int interpolation, int borderMode, const cv::Scalar& borderValue)
{
    cv::Mat result;
    cv::remap(src, result, map1, map2, interpolation, borderMode, borderValue);
    return result;
}

inline cv::Mat cvret::invertAffineTransform(cv::InputArray M)
{
    cv::Mat result;
    cv::invertAffineTransform(M, result);
    return result;
}

inline cv::Mat cvret::getRectSubPix(cv::InputArray image, cv::Size patchSize, cv::Point2f center, int patchType)
{
    cv::Mat result;
    cv::getRectSubPix(image, patchSize, center, result, patchType);
    return result;
}

inline cv::Mat cvret::integral(cv::InputArray src, int sdepth)
{
    cv::Mat result;
    cv::integral(src, result, sdepth);
    return result;
}

inline cv::Mat cvret::accumulate(cv::InputArray src, cv::InputArray mask)
{
    cv::Mat result;
    cv::accumulate(src, result, mask);
    return result;
}

inline cv::Mat cvret::accumulateSquare(cv::InputArray src, cv::InputArray mask)
{
    cv::Mat result;
    cv::accumulateSquare(src, result, mask);
    return result;
}

inline cv::Mat cvret::accumulateProduct(cv::InputArray src1, cv::InputArray src2, cv::InputArray mask)
{
    cv::Mat result;
    cv::accumulateProduct(src1, src2, result, mask);
    return result;
}

inline cv::Mat cvret::accumulateWeighted(cv::InputArray src, double alpha, cv::InputArray mask)
{
    cv::Mat result;
    cv::accumulateWeighted(src, result, alpha, mask);
    return result;
}

inline cv::Mat cvret::threshold(cv::InputArray src, double thresh, double maxval, int type)
{
    cv::Mat result;
    cv::threshold(src, result, thresh, maxval, type);
    return result;
}

inline cv::Mat cvret::adaptiveThreshold(cv::InputArray src, double maxValue, int adaptiveMethod, int thresholdType, int blockSize, double C)
{
    cv::Mat result;
    cv::adaptiveThreshold(src, result, maxValue, adaptiveMethod, thresholdType, blockSize, C);
    return result;
}

inline cv::Mat cvret::pyrDown(cv::InputArray src, const cv::Size& dstsize, int borderType)
{
    cv::Mat result;
#if CV_MAJOR_VERSION == 2 && CV_MINOR_VERSION >= 4
    cv::pyrDown(src, result, dstsize, borderType);
#else
    cv::pyrDown(src, result, dstsize);
#endif
    return result;
}

inline cv::Mat cvret::pyrUp(cv::InputArray src, const cv::Size& dstsize, int borderType)
{
    cv::Mat result;
#if CV_MAJOR_VERSION == 2 && CV_MINOR_VERSION >= 4
    cv::pyrUp(src, result, dstsize, borderType);
#else
    cv::pyrUp(src, result, dstsize);
#endif
    return result;
}

inline cv::Mat cvret::undistort(cv::InputArray src, cv::InputArray cameraMatrix, cv::InputArray distCoeffs, cv::InputArray newcameraMatrix)
{
    cv::Mat result;
    cv::undistort(src, result, cameraMatrix, distCoeffs, newcameraMatrix);
    return result;
}

inline cv::Mat cvret::calcHist(cv::InputArrayOfArrays images, const std::vector<int>& channels, cv::InputArray mask, const std::vector<int>& histSize, const std::vector<float>& ranges, bool accumulate)
{
    cv::Mat result;
    cv::calcHist(images, channels, mask, result, histSize, ranges, accumulate);
    return result;
}

inline cv::Mat cvret::calcBackProject(cv::InputArrayOfArrays images, const std::vector<int>& channels, cv::InputArray hist, const std::vector<float>& ranges, double scale)
{
    cv::Mat result;
    cv::calcBackProject(images, channels, hist, result, ranges, scale);
    return result;
}

inline cv::Mat cvret::equalizeHist(cv::InputArray src)
{
    cv::Mat result;
    cv::equalizeHist(src, result);
    return result;
}

inline cv::Mat cvret::watershed(cv::InputArray image)
{
    cv::Mat result;
    cv::watershed(image, result);
    return result;
}

inline cv::Mat cvret::pyrMeanShiftFiltering(cv::InputArray src, double sp, double sr, int maxLevel, cv::TermCriteria termcrit)
{
    cv::Mat result;
    cv::pyrMeanShiftFiltering(src, result, sp, sr, maxLevel, termcrit);
    return result;
}

inline cv::Mat cvret::cvtColor(cv::InputArray src, int code, int dstCn)
{
    cv::Mat result;
    cv::cvtColor(src, result, code, dstCn);
    return result;
}

inline cv::Mat cvret::matchTemplate(cv::InputArray image, cv::InputArray templ, int method)
{
    cv::Mat result;
    cv::matchTemplate(image, templ, result, method);
    return result;
}

inline cv::Mat cvret::drawContours(cv::InputArrayOfArrays contours, int contourIdx, const cv::Scalar& color, int thickness, int lineType, cv::InputArray hierarchy, int maxLevel, cv::Point offset)
{
    cv::Mat result;
    cv::drawContours(result, contours, contourIdx, color, thickness, lineType, hierarchy, maxLevel, offset);
    return result;
}

inline cv::Mat cvret::approxPolyDP(cv::InputArray curve, double epsilon, bool closed)
{
    cv::Mat result;
    cv::approxPolyDP(curve, result, epsilon, closed);
    return result;
}

inline cv::Mat cvret::convexHull(cv::InputArray points, bool clockwise, bool returnPoints)
{
    cv::Mat result;
    cv::convexHull(points, result, clockwise, returnPoints);
    return result;
}

inline cv::Mat cvret::fitLine(cv::InputArray points, int distType, double param, double reps, double aeps)
{
    cv::Mat result;
    cv::fitLine(points, result, distType, param, reps, aeps);
    return result;
}

inline cv::Mat cvret::add(cv::InputArray src1, cv::InputArray src2, cv::InputArray mask, int dtype)
{
    cv::Mat result;
    cv::add(src1, src2, result, mask, dtype);
    return result;
}

inline cv::Mat cvret::subtract(cv::InputArray src1, cv::InputArray src2, cv::InputArray mask, int dtype)
{
    cv::Mat result;
    cv::subtract(src1, src2, result, mask, dtype);
    return result;
}

inline cv::Mat cvret::multiply(cv::InputArray src1, cv::InputArray src2, double scale, int dtype)
{
    cv::Mat result;
    cv::multiply(src1, src2, result, scale, dtype);
    return result;
}

inline cv::Mat cvret::divide(cv::InputArray src1, cv::InputArray src2, double scale, int dtype)
{
    cv::Mat result;
    cv::divide(src1, src2, result, scale, dtype);
    return result;
}

inline cv::Mat cvret::divide(double scale, cv::InputArray src2, int dtype)
{
    cv::Mat result;
    cv::divide(scale, src2, result, dtype);
    return result;
}

inline cv::Mat cvret::scaleAdd(cv::InputArray src1, double alpha, cv::InputArray src2)
{
    cv::Mat result;
    cv::scaleAdd(src1, alpha, src2, result);
    return result;
}

inline cv::Mat cvret::addWeighted(cv::InputArray src1, double alpha, cv::InputArray src2, double beta, double gamma, int dtype)
{
    cv::Mat result;
    cv::addWeighted(src1, alpha, src2, beta, gamma, result, dtype);
    return result;
}

inline cv::Mat cvret::convertScaleAbs(cv::InputArray src, double alpha, double beta)
{
    cv::Mat result;
    cv::convertScaleAbs(src, result, alpha, beta);
    return result;
}

inline cv::Mat cvret::normalize(cv::InputArray src, double alpha, double beta, int norm_type, int dtype, cv::InputArray mask)
{
    cv::Mat result;
    cv::normalize(src, result, alpha, beta, norm_type, dtype, mask);
    return result;
}

inline cv::Mat cvret::reduce(cv::InputArray src, int dim, int rtype, int dtype)
{
    cv::Mat result;
    cv::reduce(src, result, dim, rtype, dtype);
    return result;
}

inline cv::Mat cvret::merge(const std::vector<cv::Mat>& mv)
{
    cv::Mat result;
    cv::merge(mv, result);
    return result;
}

inline cv::Mat cvret::extractChannel(cv::InputArray src, int coi)
{
    cv::Mat result;
    cv::extractChannel(src, result, coi);
    return result;
}

inline cv::Mat cvret::insertChannel(cv::InputArray src, int coi)
{
    cv::Mat result;
    cv::insertChannel(src, result, coi);
    return result;
}

inline cv::Mat cvret::flip(cv::InputArray src, int flipCode)
{
    cv::Mat result;
    cv::flip(src, result, flipCode);
    return result;
}

inline cv::Mat cvret::repeat(cv::InputArray src, int ny, int nx)
{
    cv::Mat result;
    cv::repeat(src, ny, nx, result);
    return result;
}

inline cv::Mat cvret::hconcat(cv::InputArray src)
{
    cv::Mat result;
    cv::hconcat(src, result);
    return result;
}

inline cv::Mat cvret::vconcat(cv::InputArray src)
{
    cv::Mat result;
    cv::vconcat(src, result);
    return result;
}

inline cv::Mat cvret::hconcat(cv::InputArray src1, cv::InputArray src2)
{
    cv::Mat result;
    cv::hconcat(src1, src2, result);
    return result;
}

inline cv::Mat cvret::vconcat(cv::InputArray src1, cv::InputArray src2)
{
    cv::Mat result;
    cv::vconcat(src1, src2, result);
    return result;
}

inline cv::Mat cvret::bitwise_and(cv::InputArray src1, cv::InputArray src2, cv::InputArray mask)
{
    cv::Mat result;
    cv::bitwise_and(src1, src2, result, mask);
    return result;
}

inline cv::Mat cvret::bitwise_or(cv::InputArray src1, cv::InputArray src2, cv::InputArray mask)
{
    cv::Mat result;
    cv::bitwise_or(src1, src2, result, mask);
    return result;
}

inline cv::Mat cvret::bitwise_xor(cv::InputArray src1, cv::InputArray src2, cv::InputArray mask)
{
    cv::Mat result;
    cv::bitwise_xor(src1, src2, result, mask);
    return result;
}

inline cv::Mat cvret::bitwise_not(cv::InputArray src, cv::InputArray mask)
{
    cv::Mat result;
    cv::bitwise_not(src, result, mask);
    return result;
}

inline cv::Mat cvret::absdiff(cv::InputArray src1, cv::InputArray src2)
{
    cv::Mat result;
    cv::absdiff(src1, src2, result);
    return result;
}

inline cv::Mat cvret::inRange(cv::InputArray src, cv::InputArray lowerb, cv::InputArray upperb)
{
    cv::Mat result;
    cv::inRange(src, lowerb, upperb, result);
    return result;
}

inline cv::Mat cvret::compare(cv::InputArray src1, cv::InputArray src2, int cmpop)
{
    cv::Mat result;
    cv::compare(src1, src2, result, cmpop);
    return result;
}

inline cv::Mat cvret::sqrt(cv::InputArray src)
{
    cv::Mat result;
    cv::sqrt(src, result);
    return result;
}

inline cv::Mat cvret::pow(cv::InputArray src, double power)
{
    cv::Mat result;
    cv::pow(src, power, result);
    return result;
}

inline cv::Mat cvret::exp(cv::InputArray src)
{
    cv::Mat result;
    cv::exp(src, result);
    return result;
}

inline cv::Mat cvret::log(cv::InputArray src)
{
    cv::Mat result;
    cv::log(src, result);
    return result;
}

inline cv::Mat cvret::phase(cv::InputArray x, cv::InputArray y, bool angleInDegrees)
{
    cv::Mat result;
    cv::phase(x, y, result, angleInDegrees);
    return result;
}

inline cv::Mat cvret::magnitude(cv::InputArray x, cv::InputArray y)
{
    cv::Mat result;
    cv::magnitude(x, y, result);
    return result;
}

inline cv::Mat cvret::gemm(cv::InputArray src1, cv::InputArray src2, double alpha, cv::InputArray src3, double gamma, int flags)
{
    cv::Mat result;
    cv::gemm(src1, src2, alpha, src3, gamma, result, flags);
    return result;
}

inline cv::Mat cvret::mulTransposed(cv::InputArray src, bool aTa, cv::InputArray delta, double scale, int dtype)
{
    cv::Mat result;
    cv::mulTransposed(src, result, aTa, delta, scale, dtype);
    return result;
}

inline cv::Mat cvret::transpose(cv::InputArray src)
{
    cv::Mat result;
    cv::transpose(src, result);
    return result;
}

inline cv::Mat cvret::transform(cv::InputArray src, cv::InputArray m)
{
    cv::Mat result;
    cv::transform(src, result, m);
    return result;
}

inline cv::Mat cvret::perspectiveTransform(cv::InputArray src, cv::InputArray m)
{
    cv::Mat result;
    cv::perspectiveTransform(src, result, m);
    return result;
}

inline cv::Mat cvret::completeSymm(bool lowerToUpper)
{
    cv::Mat result;
    cv::completeSymm(result, lowerToUpper);
    return result;
}

inline cv::Mat cvret::setIdentity(const cv::Scalar& s)
{
    cv::Mat result;
    cv::setIdentity(result, s);
    return result;
}

inline cv::Mat cvret::invert(cv::InputArray src, int flags)
{
    cv::Mat result;
    cv::invert(src, result, flags);
    return result;
}

inline cv::Mat cvret::solve(cv::InputArray src1, cv::InputArray src2, int flags)
{
    cv::Mat result;
    cv::solve(src1, src2, result, flags);
    return result;
}

inline cv::Mat cvret::sort(cv::InputArray src, int flags)
{
    cv::Mat result;
    cv::sort(src, result, flags);
    return result;
}

inline cv::Mat cvret::sortIdx(cv::InputArray src, int flags)
{
    cv::Mat result;
    cv::sortIdx(src, result, flags);
    return result;
}

inline cv::Mat cvret::solveCubic(cv::InputArray coeffs)
{
    cv::Mat result;
    cv::solveCubic(coeffs, result);
    return result;
}

inline cv::Mat cvret::solvePoly(cv::InputArray coeffs, int maxIters)
{
    cv::Mat result;
    cv::solvePoly(coeffs, result, maxIters);
    return result;
}

inline cv::Mat cvret::PCAProject(cv::InputArray data, cv::InputArray mean, cv::InputArray eigenvectors)
{
    cv::Mat result;
    cv::PCAProject(data, mean, eigenvectors, result);
    return result;
}

inline cv::Mat cvret::PCABackProject(cv::InputArray data, cv::InputArray mean, cv::InputArray eigenvectors)
{
    cv::Mat result;
    cv::PCABackProject(data, mean, eigenvectors, result);
    return result;
}

inline cv::Mat cvret::SVBackSubst(cv::InputArray w, cv::InputArray u, cv::InputArray vt, cv::InputArray rhs)
{
    cv::Mat result;
    cv::SVBackSubst(w, u, vt, rhs, result);
    return result;
}

inline cv::Mat cvret::dft(cv::InputArray src, int flags, int nonzeroRows)
{
    cv::Mat result;
    cv::dft(src, result, flags, nonzeroRows);
    return result;
}

inline cv::Mat cvret::idft(cv::InputArray src, int flags, int nonzeroRows)
{
    cv::Mat result;
    cv::idft(src, result, flags, nonzeroRows);
    return result;
}

inline cv::Mat cvret::dct(cv::InputArray src, int flags)
{
    cv::Mat result;
    cv::dct(src, result, flags);
    return result;
}

inline cv::Mat cvret::idct(cv::InputArray src, int flags)
{
    cv::Mat result;
    cv::idct(src, result, flags);
    return result;
}

inline cv::Mat cvret::mulSpectrums(cv::InputArray a, cv::InputArray b, int flags, bool conjB)
{
    cv::Mat result;
    cv::mulSpectrums(a, b, result, flags, conjB);
    return result;
}

inline cv::Mat cvret::fillConvexPoly(cv::InputArray points, const cv::Scalar& color, int lineType, int shift)
{
    cv::Mat result;
    cv::fillConvexPoly(result, points, color, lineType, shift);
    return result;
}

inline cv::Mat cvret::fillPoly(cv::InputArrayOfArrays pts, const cv::Scalar& color, int lineType, int shift, cv::Point offset)
{
    cv::Mat result;
    cv::fillPoly(result, pts, color, lineType, shift, offset);
    return result;
}

inline cv::Mat cvret::polylines(cv::InputArrayOfArrays pts, bool isClosed, const cv::Scalar& color, int thickness, int lineType, int shift)
{
    cv::Mat result;
    cv::polylines(result, pts, isClosed, color, thickness, lineType, shift);
    return result;
}


#endif // CVRET_HPP
