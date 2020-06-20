#ifndef CVENUMS_HPP
#define CVENUMS_HPP

#include <opencv2/opencv.hpp>

#if CV_MAJOR_VERSION == 2 && CV_MINOR_VERSION <= 4
namespace cv
{

enum {
    REDUCE_AVG = CV_REDUCE_AVG,
    REDUCE_SUM = CV_REDUCE_SUM,
    REDUCE_MIN = CV_REDUCE_MIN,
    REDUCE_MAX = CV_REDUCE_MAX,
};

}
#endif // CV_MAJOR_VERSION == 2 && CV_MINOR_VERSION <= 4

#if CV_MAJOR_VERSION == 2 && CV_MINOR_VERSION == 3
namespace cv
{

enum {
    IMREAD_GRAYSCALE = CV_LOAD_IMAGE_GRAYSCALE
};

}
#endif // CV_MAJOR_VERSION == 2 && CV_MINOR_VERSION == 3

#endif // CVENUMS_HPP
