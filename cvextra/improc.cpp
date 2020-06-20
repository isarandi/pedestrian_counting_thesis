#include "improc.hpp"
#include "vectors.hpp"
#include "cvret.hpp"
#include <opencv2/opencv.hpp>
#include <omp.h>
#include "cairoConvert.hpp"
#include "mats.hpp"
#include "colors.hpp"

using namespace std;
using namespace cv;
using namespace cvx;

auto cvx::
getGaussianKernel(Size ksize, double sigmaX, double sigmaY, int ktype) -> Mat
{
    Mat gaussKernelX = cv::getGaussianKernel(ksize.width, sigmaX, CV_64F).t();
    Mat gaussKernelY = cv::getGaussianKernel(ksize.height, sigmaY, CV_64F);

    Mat1d impulse{ksize, 0.0};
    impulse.at<double>(ksize.height/2, ksize.width/2) = 1.0;

    return cvret::sepFilter2D(
            impulse,
            ktype,
            gaussKernelX,
            gaussKernelY,
            Point{-1,-1},
            0,
            BORDER_CONSTANT);
}

void cvx::
resizeBest(InputArray src, OutputArray dst, Size desiredSize, double fx, double fy)
{
    cv::resize(
            src,
            dst,
            desiredSize,
            0,0,
            (src.size().area() > desiredSize.area()) ? INTER_AREA : INTER_CUBIC);
}

auto cvxret::
resizeBest(InputArray src, Size desiredSize, double fx, double fy) -> Mat
{
    Mat result;
    cvx::resizeBest(src, result, desiredSize);
    return result;
}

auto cvxret::
resizeByRatio(InputArray src, double factor, int interpolation) -> Mat
{
    Size targetSize{
        static_cast<int>(src.size().width*factor),
        static_cast<int>(src.size().height*factor)};

    return interpolation == -1 ?
                cvxret::resizeBest(src, targetSize, factor, factor) :
                cvret::resize(src, targetSize, factor, factor,interpolation);
}

void cvx::
rotate(
        InputArray src,
        OutputArray dst,
        Point2d anchor,
        double angle,
        AngleUnit angleUnit,
        int flags,
        int borderMode,
        Scalar borderValue)
{
    double angleInDegrees =
            (angleUnit == AngleUnit::DEGREES ? angle : angle*180.0/CV_PI);

    cv::Mat rotMatrix =
            cv::getRotationMatrix2D(
                {(float)anchor.x, (float)anchor.y},
                angleInDegrees, 1.0);

    cv::warpAffine(src, dst, rotMatrix, src.size(), flags, borderMode, borderValue);
}

template <typename T, int D>
void cvx::
lineProfile(
        cv::InputArray src,
        cv::OutputArray dst,
        cvx::LineSegment const& segment)
{
	LineSegmentProperties prop = segment.properties();
    int nSamplesAlongLine = static_cast<int>(prop.length);
    dst.create(1, nSamplesAlongLine, src.type());
    cv::Mat profile = dst.getMat();

    for (int iSample : cvx::irange(nSamplesAlongLine))
    {
        cv::Point2d p = prop.localToGlobal(iSample);
        profile.at<cv::Vec<T, D>>(0,iSample) =
                cvx::bilinear<T, D>(src, p, cv::BORDER_DEFAULT);
    }
}

void cvx::
putTextCentered(
        Mat& image,
        string const& text,
        Point2d center,
        int fontFace,
        double fontScale,
        Scalar color)
{
    int baseline_dummy;
    Size textSize = cv::getTextSize(text, fontFace, fontScale, 1, &baseline_dummy);

    double targetMidX = center.x;
    double targetMidY = center.y;

    int targetLeftX = static_cast<int>(targetMidX-textSize.width/2.0);
    int targetBottomY = static_cast<int>(targetMidY+textSize.height/2.0);

    cv::putText(
            image,
            text,
            Point{targetLeftX, targetBottomY},
            fontFace,
            fontScale,
            color,
            1,
            CV_AA);
}

void cvx::
putTextCairo(
        Mat &targetImage,
        string const& text,
        Point2d centerAnchor,
        string const& fontFamily,
        double fontSize,
        Scalar textColor,
        bool fontItalic,
        bool fontBold)
{
    // Create Cairo
    cairo_surface_t* surface =
            cairo_image_surface_create(
                CAIRO_FORMAT_ARGB32,
                targetImage.cols,
                targetImage.rows);

    cairo_t* cairo = cairo_create(surface);

    // Wrap Cairo with a Mat
    Mat cairoTarget(
                cairo_image_surface_get_height(surface),
                cairo_image_surface_get_width(surface),
                CV_8UC4,
                cairo_image_surface_get_data(surface),
                cairo_image_surface_get_stride(surface));

    // Put image onto Cairo
    cv::cvtColor(targetImage, cairoTarget, COLOR_BGR2BGRA);

    // Set font and write text
    cairo_select_font_face (
                cairo,
                fontFamily.c_str(),
                fontItalic ? CAIRO_FONT_SLANT_ITALIC : CAIRO_FONT_SLANT_NORMAL,
                fontBold ? CAIRO_FONT_WEIGHT_BOLD : CAIRO_FONT_WEIGHT_NORMAL);

    cairo_set_font_size(cairo, fontSize);
    cairo_set_source_rgb(cairo, textColor[2], textColor[1], textColor[0]);

    cairo_text_extents_t extents;
    cairo_text_extents(cairo, text.c_str(), &extents);

    cairo_move_to(
                cairo,
                centerAnchor.x - extents.width/2 - extents.x_bearing,
                centerAnchor.y - extents.height/2- extents.y_bearing);
    cairo_show_text(cairo, text.c_str());

    // Copy the image with the text to the target
    cv::cvtColor(cairoTarget, targetImage, COLOR_BGRA2BGR);

    cairo_destroy(cairo);
    cairo_surface_destroy(surface);

}

template<typename T>
auto cvx::
bilinear(
        cv::InputArray src,
        cv::Point2d location,
        int borderType
        ) -> T
{
    CV_Assert(!src.empty());
    CV_Assert(src.channels() == 1);

    int x = static_cast<int>(std::floor(location.x));
    int y = static_cast<int>(std::floor(location.y));

    cv::Mat img = src.getMat();

    int x0 = cv::borderInterpolate(x,   img.cols, borderType);
    int x1 = cv::borderInterpolate(x+1, img.cols, borderType);
    int y0 = cv::borderInterpolate(y,   img.rows, borderType);
    int y1 = cv::borderInterpolate(y+1, img.rows, borderType);

    double a = location.x - static_cast<double>(x);
    double b = 1.0-a;
    double c = location.y - static_cast<double>(y);
    double d = 1.0-c;

    return cv::saturate_cast<T>(
                (img.at<T>(y0,x0) * b + img.at<T>(y0,x1) * a) * d +
                (img.at<T>(y1,x0) * b + img.at<T>(y1,x1) * a) * c);
}

template<typename T, int D>
auto cvx::
bilinear(
        InputArray src,
        Point2d location,
        int borderType
        ) -> Vec<T,D>
{
    CV_Assert(!src.empty());
    CV_Assert(src.channels() == D);

    int x = static_cast<int>(std::floor(location.x));
    int y = static_cast<int>(std::floor(location.y));

    Mat img = src.getMat();

    int x0 = cv::borderInterpolate(x,   img.cols, borderType);
    int x1 = cv::borderInterpolate(x+1, img.cols, borderType);
    int y0 = cv::borderInterpolate(y,   img.rows, borderType);
    int y1 = cv::borderInterpolate(y+1, img.rows, borderType);

    double a = location.x - static_cast<double>(x);
    double b = 1.0-a;
    double c = location.y - static_cast<double>(y);
    double d = 1.0-c;

    Vec<T,D> result;
    for (int i=0; i < D; ++i)
    {
        result[i] = cv::saturate_cast<T>(
                    (img.at<cv::Vec<T,D>>(y0,x0)[i] * b + img.at<cv::Vec<T,D>>(y0,x1)[i] * a) * d +
                    (img.at<cv::Vec<T,D>>(y1,x0)[i] * b + img.at<cv::Vec<T,D>>(y1,x1)[i] * a) * c);
    }
    return result;
}


void cvx::
grid(Mat& dst, Size gridResolution, Scalar color, int thickness)
{
    double cellWidth = dst.cols / static_cast<double>(gridResolution.width);
    double cellHeight = dst.rows / static_cast<double>(gridResolution.height);

    // horizontal lines
    for (int row = 1; row < gridResolution.height; ++row)
    {
        cv::line(
                dst,
                Point{0, (int)(row*cellHeight)},
                Point{dst.cols, (int)(row*cellHeight)},
                color,
                thickness);
    }

    // vertical lines
    for (int col = 1; col < gridResolution.width; ++col)
    {
        cv::line(
                dst,
                Point{(int)(col*cellWidth), 0},
                Point{(int)(col*cellWidth), dst.rows},
                color,
                thickness);
    }
}

auto cvxret::
fillSmallHoles(
        BinaryMat const& binary,
        int minSizeToKeep,
        Connectivity connectivity
        ) -> BinaryMat
{
    return 255-removeSmallConnectedComponents(255-binary, minSizeToKeep, connectivity);
}

auto cvxret::
outline(InputArray src) -> BinaryMat
{
    Mat disk = cv::getStructuringElement(MORPH_ELLIPSE, Size{3,3});
    Mat dilated = cvret::dilate(src, disk);

    return dilated - src.getMat();
}

auto cvxret::
removeSmallConnectedComponents(
        BinaryMat const& binary,
        int minSizeToKeep,
        Connectivity connectivity
        ) -> BinaryMat
{
    Mat1b labels = binary.clone();
    int floodFillFlags = (connectivity == Connectivity::FOUR ? 4 : 8);

    // Label used when flood filling the current component (whose size is not yet known)
    int tempLabel = 2;

    for (Point componentReferencePoint : cvx::points(labels))
    {
        if (labels(componentReferencePoint) == 255)
        {
            Rect rect;
            cv::floodFill(
                    labels,
                    componentReferencePoint,
                    Scalar(tempLabel),
                    &rect,
                    Scalar(),
                    Scalar(),
                    floodFillFlags);

            int nPixelsInComponent = 0;
            for (Point p : cvx::points(rect))
            {
                if (labels(p) == tempLabel)
                {
                    ++nPixelsInComponent;
                }
            }

            bool componentLargeEnough = (nPixelsInComponent >= minSizeToKeep);
            for (Point p : cvx::points(rect))
            {
                if (labels(p) == tempLabel)
                {
                    // We set it to 1 instead of 255, so that we can differentiate between
                    // yet unprocessed white pixels and already size-checked component pixels
                    labels(p) = (componentLargeEnough ? 1 : 0);
                }
            }
        }
    }

    // Turn our 1's into 255's as conventional with binary images
    return cvret::threshold(labels, 0, 255, cv::THRESH_BINARY);
}

void cvx::
arrow(Mat& dst, Point2d start, Vec2d direction, Scalar color, int thickness)
{
    double angle = std::atan2(direction[1], direction[0]);

    Point tip;
    tip.x = (int) (start.x + direction[0]);
    tip.y = (int) (start.y + direction[1]);

    // Draw main line of arrow.
    cv::line(dst, {(int)start.x, (int)start.y}, tip, color, thickness, CV_AA, 0);

    // Draw the tips of the arrow.  Do some scaling so that the
    // tips look proportional to the main line of the arrow.
    Point arrowtipside;
    arrowtipside.x = (int) (tip.x - 3 * std::cos(angle + CV_PI / 4));
    arrowtipside.y = (int) (tip.y - 3 * std::sin(angle + CV_PI / 4));
    cv::line(dst, arrowtipside, tip, color, thickness, CV_AA, 0);

    arrowtipside.x = (int) (tip.x - 3 * std::cos(angle - CV_PI / 4));
    arrowtipside.y = (int) (tip.y - 3 * std::sin(angle - CV_PI / 4));
    cv::line(dst, arrowtipside, tip, color, thickness, CV_AA, 0);
}

void cvx::
findContours(InputArray image, OutputArrayOfArrays dst, int mode, int method, Point offset)
{
    Mat copy = image.getMat().clone();
    cv::findContours(copy, dst, mode, method, offset);
}

auto cvxret::
findContours(
        InputArray image,
        int mode,
        int method,
        Point offset
        ) -> vector< vector<Point> >
{
    vector< vector<Point> > result;
    cvx::findContours(image, result, mode, method, offset);
    return result;
}

auto cvxret::
rotate(
        InputArray src,
        Point2d anchor,
        double angle,
        AngleUnit angleUnit,
        int flags,
        int borderMode,
        Scalar borderValue
        ) -> Mat
{
    Mat result;
    cvx::rotate(src, result, anchor, angle, angleUnit, flags, borderMode, borderValue);
    return result;
}


auto cvxret::
skeleton(InputArray src) -> BinaryMat
{
    Mat img = src.getMat().clone();

    BinaryMat skel = Mat1b::zeros(img.size());
    Mat temp;
    Mat eroded;

    Mat element = cv::getStructuringElement(MORPH_CROSS, {3, 3});

    do
    {
        cv::erode(img, eroded, element);
        cv::dilate(eroded, temp, element); // temp = open(img)
        cv::subtract(img, temp, temp);
        cv::bitwise_or(skel, temp, skel);
        eroded.copyTo(img);

    } while (cv::countNonZero(img) != 0);

    return skel;
}

auto cvxret::
gradientField(Mat1d const& mat) -> Mat2d
{
    return cvret::merge({
            cvret::Sobel(mat, CV_64F, 1, 0),
            cvret::Sobel(mat, CV_64F, 0, 1)});
}

void cvx::
gradientField(InputArray src1d, OutputArray dst2d)
{
    assert(src1d.channels() == 1 && dst2d.channels() == 2);

    cv::merge({cvret::Sobel(src1d, CV_64F, 1, 0),
              cvret::Sobel(src1d, CV_64F, 0, 1)},
              dst2d);
}

void cvx::setChannel(InputArray src, InputOutputArray dst, int channelIdx)
{
    auto channels = cvret::split(dst.getMat());
    channels[channelIdx] = src.getMat();
    cv::merge(channels, dst);
}

void cvx::
lineProfile(InputArray src, OutputArray dst, LineSegment const& segment)
{
    switch (src.depth())
    {
    case CV_64F:
        switch (src.channels()) {
        case 1:
            lineProfile<double,1>(src, dst, segment); break;
        case 2:
            lineProfile<double,2>(src, dst, segment); break;
        case 3:
            lineProfile<double,3>(src, dst, segment); break;
        case 4:
            lineProfile<double,4>(src, dst, segment); break;
        case 5:
            lineProfile<double,5>(src, dst, segment); break;
        } break;
    case CV_8U:
        switch (src.channels()) {
        case 1:
            lineProfile<uchar,1>(src, dst, segment); break;
        case 2:
            lineProfile<uchar,2>(src, dst, segment); break;
        case 3:
            lineProfile<uchar,3>(src, dst, segment); break;
        case 4:
            lineProfile<uchar,4>(src, dst, segment); break;
        case 5:
            lineProfile<uchar,5>(src, dst, segment); break;
        } break;
    case CV_32F:
        switch (src.channels()) {
        case 1:
            lineProfile<float,1>(src, dst, segment); break;
        case 2:
            lineProfile<float,2>(src, dst, segment); break;
        case 3:
            lineProfile<float,3>(src, dst, segment); break;
        case 4:
            lineProfile<float,4>(src, dst, segment); break;
        case 5:
            lineProfile<float,5>(src, dst, segment); break;
        } break;
    case CV_32S:
        switch (src.channels()) {
        case 1:
            lineProfile<int,1>(src, dst, segment); break;
        case 2:
            lineProfile<int,2>(src, dst, segment); break;
        case 3:
            lineProfile<int,3>(src, dst, segment); break;
        case 4:
            lineProfile<int,4>(src, dst, segment); break;
        case 5:
            lineProfile<int,5>(src, dst, segment); break;
        } break;
    }
}

auto cvxret::
lineProfile(InputArray src, LineSegment const& seg) -> Mat
{
    Mat result;
    cvx::lineProfile(src, result, seg);
    return result;
}

// instantiate
#define INSTANTIATE_BILINEAR(T) \
    template \
    auto cvx::bilinear<T>( \
            cv::InputArray src, \
            cv::Point2d location, \
            int borderType \
            ) -> T;

INSTANTIATE_BILINEAR(double)
INSTANTIATE_BILINEAR(float)

auto cvx::
boundingBoxOfBinary(cv::InputArray src) -> Rect
{
    Point tl{-1,-1};
    Point br{-1,-1};

    Mat1b mat = src.getMat();

    for (Point const& p : cvx::points(mat))
    {
        if (mat(p))
        {
            if (tl.x == -1 || p.x < tl.x)
            {
                tl.x = p.x;
            }
            if (tl.y == -1 || p.y < tl.y)
            {
                tl.y = p.y;
            }

            br.x = std::max(br.x, p.x);
            br.y = std::max(br.y, p.y);
        }
    }

    if (tl == Point{-1,-1})
    {
        return Rect{0,0,0,0};
    }

    return Rect{tl, Size{br.x-tl.x+1, br.y-tl.y+1}};
}

auto cvxret::
cannyPlus(
		const cv::InputArray& src,
		double gaussRadius,
		double threshold1,
		double threshold2,
		int minComponentSize
		) -> cvx::BinaryMat
{
    auto blurred =
            cvret::GaussianBlur(
                    src,
                    {11,11},
					gaussRadius,
					gaussRadius);

    auto canny = cvret::Canny(blurred, threshold1, threshold2);
    auto filtered =
            cvxret::removeSmallConnectedComponents(
                    canny,
                    minComponentSize,
                    cvx::Connectivity::EIGHT);

    return filtered;
}
