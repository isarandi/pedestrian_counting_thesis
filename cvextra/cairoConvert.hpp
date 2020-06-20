#ifndef CAIROCONVERT_HPP
#define CAIROCONVERT_HPP


#include <opencv2/opencv.hpp>
#include <cairo/cairo.h>

#include <iostream>
#include <vector>
#include <memory>

namespace cvx { namespace cairo {


class CairoImage{
public:

    ~CairoImage()
    {
        cairo_destroy(cairo);
        cairo_surface_destroy(surface);
    }

    cairo_surface_t* surface;
    cairo_t* cairo;


};

auto matToCairo(cv::Mat const& matSource) -> std::unique_ptr<CairoImage>;
auto cairoToMat(CairoImage const& cairoImage) -> cv::Mat;


void putText(
        CairoImage &im,
        std::string const& text,
        cv::Point2d centerAnchor,
        std::string const& fontFamily,
        cairo_font_slant_t fontSlant,
        cairo_font_weight_t fontWeight,
        double fontSize,
        cv::Scalar color);

}}

#endif // CAIROCONVERT_HPP
