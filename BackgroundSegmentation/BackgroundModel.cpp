#include <BackgroundSegmentation/BackgroundModel.hpp>
#include <opencv2/core/core.hpp>

using namespace std;
using namespace cv;
using namespace cvx;
using namespace crowd::bg;

BinaryMat BackgroundModel::segmentNextRet(InputArray nextImage)
{
    BinaryMat result;
    segmentNext(nextImage, result);
    return result;
}

