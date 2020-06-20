#include <CrowdCounting/RegionCounting/CountingFrame.hpp>
#include <CrowdCounting/CrowdCountingUtils.hpp>

#include <boost/filesystem.hpp>
#include <cvextra/coords.hpp>
#include <cvextra/core.hpp>
#include <cvextra/cvret.hpp>
#include <cvextra/LoopRange.hpp>
#include <cvextra/cvenums.hpp>
#include <cvextra/filesystem.hpp>
#include <cvextra/improc.hpp>
#include <cvextra/io.hpp>

#include <opencv2/opencv.hpp>
#include <cassert>
#include <string>

using namespace std;
using namespace cv;
using namespace cvx;
using namespace crowd;

CountingFrame::
CountingFrame(
        bpath const& framePath,
        bpath const& maskPath,
		bpath const& textonPath,
        vector<Point2d> const& peoplePositions,
        Size processingSize,
        Mat const& scaleMap,
        BinaryMat const& roiMask)
    : framePath(framePath)
    , maskPath(maskPath)
	, textonMapPath(textonPath)
    , processingSize(processingSize)
    , scaleMap(scaleMap)
    , roiMask(roiMask)
{
    std::copy_if(
            peoplePositions.begin(),
            peoplePositions.end(),
            std::back_inserter(this->peoplePositions),
            [this](Point2d const& pos){
                return cvx::atRel(this->roiMask, pos) != 0;});
}

auto CountingFrame::
getFrame() const -> Mat
{
    Mat frame = cvx::imread(framePath);
    cvx::resizeBest(frame, frame, processingSize);
    return frame;
}

auto CountingFrame::
getFullResolutionFrame() const -> Mat
{
    return cvx::imread(framePath);
}

auto CountingFrame::
getMask() const -> BinaryMat
{
    BinaryMat mask = cvx::imread(maskPath, IMREAD_GRAYSCALE);
    cvx::resizeBest(mask, mask, processingSize);
    cv::threshold(mask, mask, 240, 255, THRESH_BINARY);
    return cv::min(mask, roiMask);
}

auto CountingFrame::
getScaleMap() const -> Mat1d
{
    return scaleMap;
}

auto CountingFrame::
getTextonMap() const -> Mat1b
{
    Mat1b textonMap = cvx::imread(textonMapPath, IMREAD_GRAYSCALE)+1;
    cvx::resizeBest(textonMap, textonMap, processingSize, INTER_NEAREST);
    return cv::min(textonMap, roiMask);
}

void CountingFrame::
saveVariant(string const& variantName, InputArray variant) const
{
    bpath variantPath =
            (framePath.parent_path().parent_path() / variantName / framePath.filename())
            .replace_extension(".png");

    cvx::imwrite(variantPath, variant);
}

auto CountingFrame::
getPeoplePositions() const -> std::vector<Point2d>
{
    return peoplePositions;
}

auto CountingFrame::
countPeopleInRectangle(Rectd relativeRectangle) const -> double
{
    double radiusX = 10/960.0;
    double radiusY = 10/540.0;
    double count = 0;

    for (Point2d pos : peoplePositions)
    {
        double integralInsideX = 1.0
                -cvx::math::standardNormalCdf((pos.x-relativeRectangle.br().x)/radiusX)
                -cvx::math::standardNormalCdf((relativeRectangle.tl().x-pos.x)/radiusX);

        double integralInsideY = 1.0
                -cvx::math::standardNormalCdf((relativeRectangle.tl().y-pos.y)/radiusY)
                -cvx::math::standardNormalCdf((pos.y-relativeRectangle.br().y)/radiusY);

        count += integralInsideX*integralInsideY;
    }

    return count;
}

Size CountingFrame::
getProcessingSize() const
{
    return processingSize;
}

auto CountingFrame::
allFromFolder(
        bpath const& framesPath,
        bpath const& maskPath,
		bpath const& textonPath,
        vector<vector<Point2d>> peoplePositions,
        Size processingSize,
        Mat scaleMap,
        BinaryMat roi
        ) -> vector<CountingFrame>
{
    if (roi.empty())
    {
        roi = Mat1b{processingSize, 255};
    }
    Mat roiProcSized =
    		(roi.size() == processingSize) ?
    				roi :
					cvret::resize(roi, processingSize, 0,0, INTER_LINEAR);
    Mat scaleMapProcSized =
    		(scaleMap.size() == processingSize) ?
    				scaleMap :
					cvret::resize(scaleMap, processingSize, 0,0, INTER_LINEAR);

    auto framePaths = cvx::filesystem::listdir(framesPath);
    auto maskPaths = cvx::filesystem::listdir(maskPath, ".*\\.png");

    auto textonPaths = cvx::filesystem::listdir(textonPath, ".*\\.png");

//    cout << framePaths.size() << endl;
//    cout << maskPaths.size() << endl;
//    cout << peoplePositions.size() << endl;
//    cout << textonPaths.size() << endl;

    assert(framePaths.size() == maskPaths.size()
           && framePaths.size() == peoplePositions.size()
		   && framePaths.size() == textonPaths.size());

    vector<CountingFrame> result;
    for (int iFrame : cvx::irange(framePaths.size()))
    {
        result.push_back(
        		CountingFrame{
					 framePaths[iFrame],
					 maskPaths[iFrame],
					 textonPaths[iFrame],
					 peoplePositions[iFrame],
					 processingSize,
					 scaleMapProcSized,
					 roiProcSized});
    }

    return result;
}
