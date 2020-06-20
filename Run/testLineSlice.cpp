#include <boost/iterator/iterator_facade.hpp>
#include <boost/uuid/uuid.hpp>
#include <cvextra/colors.hpp>
#include <cvextra/cvret.hpp>
#include <cvextra/gui.hpp>
#include <cvextra/ImageLoadingIterable.hpp>
#include <cvextra/math.hpp>
#include <cvextra/volumetric.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>
#include <string>

using namespace std;
using namespace cv;
using namespace cvx;
using namespace cvx::math;

auto guiGetLineSegment(Mat image) -> LineSegment
{
    string winname = "Line reader";
    cv::namedWindow(winname);
    cv::imshow(winname, image);

    LineSegment seg{{0,0},{0,0}};

    bool dragging = false;
    gui::addMouseEventHandler(winname, [&](int event, Point p, int){
        if (event == cv::EVENT_LBUTTONDOWN)
        {
            seg.p1 = p;
            dragging = true;
        } else if (dragging)
        {
        	seg.p2 = p;
        }

        if (event == cv::EVENT_LBUTTONUP)
        {
            dragging = false;
        }

        cv::imshow(winname, cvret::line(image, seg.p1, seg.p2, cvx::RED));
    });

    cv::waitKey();

    gui::removeMouseEventHandlers(winname);
    cv::destroyWindow(winname);

    cout << seg.p1 << ", " << seg.p2 << endl;

    return seg;
}


void testLineSlice()
{
    auto imIterable = cvx::imagesIn("/work/mehner/crange_ausschnitt1");
    Mat3b firstIm = *(imIterable.begin());

    LineSegment seg = guiGetLineSegment(firstIm);
    auto sliceImage = cvx::timeSlice(imIterable, seg);

    cv::imshow("Line slice", sliceImage);
    cv::waitKey();
}
