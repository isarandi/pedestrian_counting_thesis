#include "illustrate.hpp"
#include "cvextra/LoopRange.hpp"
#include "cvextra/ComponentIterable.hpp"

using namespace std;
using namespace cv;
using namespace cvx;

auto illustrateComponents(BinaryMat const& img) -> Mat3b
{
    RNG rng;
    Mat3b result = Mat3b::zeros(img.size());

    for (auto const& componentMask : cvx::connectedComponentMasks(img, 8))
    {
        Scalar color(rng.uniform(0,255), rng.uniform(0, 255), rng.uniform(0, 255));
        result.setTo(color, componentMask);
    }
    return result;
}

auto illustrateLabels(Mat const& img, int nLabels) -> Mat3b
{
    RNG rng(1);
    Mat3b result = Mat3b::zeros(img.size());

    for (int iLabel : cvx::irange(0,nLabels))
    {
        Scalar color(rng.uniform(0,255), rng.uniform(0, 255), rng.uniform(0, 255));
        result.setTo(color, img == iLabel);
    }
    return result;
}

