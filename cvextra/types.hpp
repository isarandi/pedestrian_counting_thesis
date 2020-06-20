#ifndef CVEXTRA_TYPES_HPP_
#define CVEXTRA_TYPES_HPP_

#include <opencv2/core/core.hpp>

template <int CVTypeId>
struct CPPType{};

#define M_GENERATE_TYPE_STRUCT(cppType, cvType) \
        template <> struct CPPType<cvType> { typedef cppType type; };

M_GENERATE_TYPE_STRUCT(uchar, CV_8U)
M_GENERATE_TYPE_STRUCT(int, CV_32S)
M_GENERATE_TYPE_STRUCT(float, CV_32F)
M_GENERATE_TYPE_STRUCT(double, CV_64F)
M_GENERATE_TYPE_STRUCT(char, CV_8S)

#endif /* CVEXTRA_TYPES_HPP_ */
