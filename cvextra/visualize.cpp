#include "visualize.hpp"
#include "cvextra/ComponentIterable.hpp"
#include <cvextra.hpp>

using namespace std;
using namespace cv;
using namespace cvx;

auto cvx::visu::
vectorFieldAsHSV(
        Mat2d const& field,
        double saturationMagnitude
        ) -> Mat3b
{
    Mat3b illust{field.size()};

    for (Point p: cvx::points(field))
    {
        Vec2d vec = field(p);
        double angleRad = std::atan2(vec[1], vec[0]);
        double angleDeg = (angleRad+CV_PI)*180/CV_PI;

        double magnitude = cv::norm(vec);

        illust(p) = {
                cv::saturate_cast<uchar>(angleDeg/2),
                cv::saturate_cast<uchar>(magnitude*255/saturationMagnitude),
                255};
    }

    return illust;
}

auto cvx::visu::
vectorFieldAsArrows(
        Mat const& background,
        Mat2d const& field,
        int step,
        double minMagnitude
        ) -> cv::Mat
{
    Mat illust;

    switch (background.type())
    {
    case CV_8UC3:
        illust = background.clone();
        break;
    case CV_8U:
        illust = cvret::cvtColor(background, COLOR_GRAY2BGR);
        break;
    case CV_64F:
        illust = cvret::cvtColor(Mat1b{background*255}, COLOR_GRAY2BGR);
        break;
    }

    double minMagnitudeSq = cvx::sq(minMagnitude);
    for (Point p: cvx::points(illust))
    {
        Vec2d v = field(p);
        if (v.dot(v) > minMagnitudeSq)
        {
            cvx::arrow(illust, p, v, cvx::WHITE);
        }
    }

    return illust;
}


auto cvx::visu::
vectorFieldAsHSVAsBGR(
        const Mat2d &field,
        double saturationMagnitude) -> Mat3b
{
    return cvret::cvtColor(cvx::visu::vectorFieldAsHSV(field, saturationMagnitude), COLOR_HSV2BGR);
}

auto cvx::visu::
grayStretch(InputArray src) -> Mat1d
{
    double minVal;
    double maxVal;
    cv::minMaxIdx(src, &minVal, &maxVal);
    Mat result;
    src.getMat().convertTo(result, CV_32F, 1.0/(maxVal-minVal), -minVal/(maxVal-minVal));
    return result;
}


auto cvx::visu::
maskIllustration(InputArray src, InputArray mask) -> Mat
{
    Mat result;
    cvx::visu::maskIllustration(src, mask, result);
    return result;
}

void cvx::visu::
highlightOnto(InputArray orig, InputArray mask, InputOutputArray illust, Scalar color)
{
    orig.getMat().copyTo(illust, mask);
    illust.getMat().setTo(color, cvxret::outline(mask));
}

auto cvx::visu::
darkened(InputArray src) -> cv::Mat3b
{
    Mat hsvImg = cvret::cvtColor(src, COLOR_BGR2HSV);

    vector<Mat> hsvChannels = cvret::split(hsvImg);
    hsvChannels[1].setTo(0);
    Mat(hsvChannels[2]/3).copyTo(hsvChannels[2]);

    return cvret::cvtColor(cvret::merge(hsvChannels), COLOR_HSV2BGR);
}

void cvx::visu::
maskIllustration(InputArray src, InputArray _mask, OutputArray illust)
{
    Mat mask = (_mask.type() == CV_8U) ? _mask.getMat() : cvret::cvtColor(_mask, COLOR_BGR2GRAY);
    if (src.size() != mask.size())
    {
        cv::resize(mask, mask, src.size(), 0,0, INTER_NEAREST);
    }

    Mat hsvImg = cvret::cvtColor(src, COLOR_BGR2HSV);

    vector<Mat> hsvChannels = cvret::split(hsvImg);
    hsvChannels[1].setTo(0, 255-mask);
    Mat(hsvChannels[2]/3).copyTo(hsvChannels[2], 255-mask);

    cv::merge(hsvChannels, hsvImg);
    cv::cvtColor(hsvImg, illust, COLOR_HSV2BGR);

    illust.getMat().setTo(cvx::GREEN, cvxret::outline(mask));
}

auto cvx::visu::
components(BinaryMat const& img) -> Mat3b
{
    RNG rng;
    Mat3b result = Mat3b::zeros(img.size());

    for (auto const& componentMask : cvx::connectedComponentMasks(img, 8))
    {
        Scalar color(rng.uniform(0,255), rng.uniform(0, 255), rng.uniform(0, 255));
        result.setTo(color, componentMask);
    }
    return result;
}

auto cvx::visu::
labels(Mat const& img) -> Mat3b
{
    RNG rng(1);
    Mat3b result = Mat3b::zeros(img.size());

    double maxVal;
    cv::minMaxIdx(img, nullptr, &maxVal);
    int nLabels = ((int)maxVal)+1;

    for (int iLabel : cvx::irange(0,nLabels))
    {
        Scalar color(rng.uniform(0,255), rng.uniform(0, 255), rng.uniform(0, 255));
        result.setTo(color, img == iLabel);
    }
    return result;
}

