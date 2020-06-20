#include "illustCrowdFlow.hpp"

using namespace std;
using namespace cv;
using namespace cvx;


auto crowd::illust::
createAnnotatedSlice(
        vector<LineCrossing> const& crossings,
        Mat3b const& imageSlice,
        int dotRadius) -> Mat
{
    Mat illust = imageSlice.clone();

    for (LineCrossing const& crossing: crossings)
    {
        if (crossing.frameId < imageSlice.rows)
        {
            Point p = {static_cast<int>(crossing.localPos), crossing.frameId};
            cv::circle(illust, p, dotRadius, crossing.dir == CrossingDir::LEFTWARD ? cvx::RED : cvx::BLUE, -1);
        }
    }

    return illust;
}
