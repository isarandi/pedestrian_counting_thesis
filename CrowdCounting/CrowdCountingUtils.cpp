#include <cvextra/colors.hpp>
#include <cvextra/coords.hpp>
#include <cvextra/core.hpp>
#include <cvextra/io.hpp>
#include <cvextra/improc.hpp>
#include <cvextra/LoopRange.hpp>
#include <cvextra/strings.hpp>
#include <CrowdCounting/RegionCounting/CountingFrame.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/core/mat.hpp>
#include <opencv2/core/core.hpp>
#include "CrowdCountingUtils.hpp"
#include <iostream>
#include <fstream>
#include <string>
#include <vector>

using namespace std;
using namespace cv;
using namespace cvx;
using namespace crowd;

auto crowd::
gridQuantize(
        vector<cv::Point2d> const& peoplePositions,
        cv::Size gridSize
        ) -> vector<double>
{
    vector<double> result(gridSize.area());

    for (Point2d const& personPos : peoplePositions)
    {
        Point gridCell = cvx::rel2abs(personPos, gridSize);
        if (cvx::contains(gridSize, gridCell))
        {
            int iGridCell = gridCell.y*gridSize.width + gridCell.x;
            ++result[iGridCell];
        }
    }

    return result;
}

auto crowd::
gridQuantizeAll(
        vector<CountingFrame> const& countingFrames,
        cv::Size gridSize
        ) -> vector<vector<double>>
{
    vector<vector<double>> result;
    for (auto const& countingFrame : countingFrames)
    {
        result.push_back(gridQuantize(countingFrame.getPeoplePositions(), gridSize));
    }

    return result;
}

auto crowd::
createGridRectangles(Size gridSize) -> std::vector<cvx::Rectd>
{
    vector<Rectd> rectangles;
    for (Point p : cvx::points(gridSize))
    {
        Rect absRect = Rect{p, Size{1,1}};
        rectangles.push_back(cvx::abs2rel(absRect, gridSize));
    }
    return rectangles;
}

auto crowd::
readPeoplePositions(string const& filePath) -> vector<vector<Point2d> >
{
    vector<vector<Point2d>> result;
    vector<Point2d> frameResult;

    for (string const& line : cvx::io::linesOf(filePath))
    {
        int commaPos = line.find(',');

        if (commaPos != -1)
        {
            double x = std::stod(line.substr(0, commaPos));
            double y = std::stod(line.substr(commaPos+2, line.size()-(commaPos+2)));
            frameResult.push_back(Point2d{x,y});
        } else if (!frameResult.empty())
        {
            result.push_back(frameResult);
            frameResult.clear();
        }
    }

    if (!frameResult.empty())
    {
       result.push_back(frameResult);
    }

    return result;
}


auto crowd::
generateCountingIllustration(
        Mat image,
        vector<double> const& peopleCountsDesired,
        vector<double> const& peopleCountsPrediction,
        Size gridSize) -> Mat
{
    Mat illust = image.clone();
    cvx::grid(illust, gridSize, cvx::YELLOW, 2);

    Size2d cellSize = cvx::size2d(image.size()) / cvx::size2d(gridSize);

    for (Point p : cvx::points(gridSize))
    {
        int iGridCell = p.y*gridSize.width + p.x;

        cvx::putTextCairo(
                    illust,
                    cvx::str::format("%.2f", peopleCountsPrediction[iGridCell]),
					Point2d{p.x+0.5, p.y+0.25} * cellSize,
                    "arial",
                    16,
                    cvx::YELLOW);

        cvx::putTextCairo(
                    illust,
                    cvx::str::format("%.2f", peopleCountsDesired[iGridCell]),
					Point2d{p.x+0.5, p.y+0.75} * cellSize,
                    "arial",
                    16,
                    cvx::YELLOW);
    }

    return illust;
}

