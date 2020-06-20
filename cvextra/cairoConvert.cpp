#include "cairoConvert.hpp"
#include <stdx/stdx.hpp>
#include <cvextra.hpp>

using namespace cv;
using namespace std;
using namespace cvx::cairo;


unique_ptr<CairoImage> cvx::cairo::matToCairo(Mat const& matSource)
{
    auto cairoImage = stdx::make_unique<CairoImage>();
    cairoImage->surface = cairo_image_surface_create(CAIRO_FORMAT_ARGB32, matSource.cols, matSource.rows);
    cairoImage->cairo = cairo_create(cairoImage->surface);


    Mat matTarget(
                cairo_image_surface_get_height(cairoImage->surface),
                cairo_image_surface_get_width(cairoImage->surface),
                CV_8UC4,
                cairo_image_surface_get_data(cairoImage->surface),
                cairo_image_surface_get_stride(cairoImage->surface));

    cv::cvtColor(matSource, matTarget, COLOR_BGR2BGRA);

    return cairoImage;
}

cv::Mat cvx::cairo::cairoToMat(CairoImage const& cairoImage)
{
    Mat matSource(
                cairo_image_surface_get_height(cairoImage.surface),
                cairo_image_surface_get_width(cairoImage.surface),
                CV_8UC4,
                cairo_image_surface_get_data(cairoImage.surface),
                cairo_image_surface_get_stride(cairoImage.surface));

    Mat result;
    cv::cvtColor(matSource, result, COLOR_BGRA2BGR);
    return result;
}


void cvx::cairo::putText(
        CairoImage &im,
        string const& text,
        Point2d centerAnchor,
        string const& fontFamily,
        cairo_font_slant_t fontSlant,
        cairo_font_weight_t fontWeight,
        double fontSize,
        Scalar color)
{
    cairo_select_font_face (im.cairo, fontFamily.c_str(), fontSlant, fontWeight);
    cairo_set_font_size(im.cairo, fontSize);
    cairo_set_source_rgb(im.cairo, color[2], color[1], color[0]);

    cairo_text_extents_t extents;
    cairo_text_extents(im.cairo, text.c_str(), &extents);

    Point2d topLeft = centerAnchor - 0.5*Vec2d{extents.width, extents.height};
    cairo_move_to(im.cairo, topLeft.x-extents.x_bearing, topLeft.y-extents.y_bearing);

    cairo_show_text(im.cairo, text.c_str());
}
