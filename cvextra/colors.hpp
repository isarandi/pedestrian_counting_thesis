#ifndef CVEXTRA_COLORS_HPP_
#define CVEXTRA_COLORS_HPP_

#include <opencv2/core/core.hpp>

namespace cvx {

auto cvtColor(cv::Vec3b const& color, int conversionType) -> cv::Vec3b;
auto cvtColor(cv::Scalar const& color, int conversionType) -> cv::Scalar;
auto toVec3b(cv::Scalar const& color) -> cv::Vec3b;
auto toScalar(cv::Vec3b const& color) -> cv::Scalar;

cv::Scalar const WHITE(255,255,255);
cv::Scalar const SILVER(192,192,192);
cv::Scalar const GRAY(128,128,128);
cv::Scalar const BLACK(0,0,0);
cv::Scalar const RED(0,0,255);
cv::Scalar const MAROON(0,0,128);
cv::Scalar const ORANGE(0,165,255);
cv::Scalar const YELLOW(0,255,255);
cv::Scalar const OLIVE(0,128,128);
cv::Scalar const LIME(0,255,0);
cv::Scalar const GREEN(0,128,0);
cv::Scalar const AQUA(255,255,0);
cv::Scalar const TEAL(128,128,0);
cv::Scalar const BLUE(255,0,0);
cv::Scalar const NAVY(128,0,0);
cv::Scalar const FUCHSIA(255,0,255);
cv::Scalar const PURPLE(128,0,128);

cv::Vec3b const V3B_WHITE(255,255,255);
cv::Vec3b const V3B_SILVER(192,192,192);
cv::Vec3b const V3B_GRAY(128,128,128);
cv::Vec3b const V3B_BLACK(0,0,0);
cv::Vec3b const V3B_RED(0,0,255);
cv::Vec3b const V3B_MAROON(0,0,128);
cv::Vec3b const V3B_ORANGE(0,165,255);
cv::Vec3b const V3B_YELLOW(0,255,255);
cv::Vec3b const V3B_OLIVE(0,128,128);
cv::Vec3b const V3B_LIME(0,255,0);
cv::Vec3b const V3B_GREEN(0,128,0);
cv::Vec3b const V3B_AQUA(255,255,0);
cv::Vec3b const V3B_TEAL(128,128,0);
cv::Vec3b const V3B_BLUE(255,0,0);
cv::Vec3b const V3B_NAVY(128,0,0);
cv::Vec3b const V3B_FUCHSIA(255,0,255);
cv::Vec3b const V3B_PURPLE(128,0,128);

}


#endif /* CVEXTRA_COLORS_HPP_ */
