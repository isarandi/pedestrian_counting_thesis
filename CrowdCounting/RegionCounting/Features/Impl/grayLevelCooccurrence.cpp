#include "features.hpp"
#include <cvextra/core.hpp>
#include <cvextra/vectors.hpp>
#include <cvextra/LoopRange.hpp>
#include <cvextra/math.hpp>
#include <cvextra/mats.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/core/mat.hpp>
#include <cmath>
#include <limits>
#include <vector>

using namespace std;
using namespace cv;
using namespace cvx;

vector<double> crowd::GLCMFeatures(InputArray in, InputArray _mask)
{
    Mat1b img = in.getMat();
    BinaryMat mask = _mask.empty() ? Mat1b(in.size(), 255) : _mask.getMat();

    int const nGrayBins = 8;

    Mat1b quantizedImage = Mat1b::zeros(in.size());
    for (Point p : cvx::points(quantizedImage))
    {
        quantizedImage(p) = cvx::math::bin(img(p), 0, 255, nGrayBins);
    }

    vector<int> nPairs(4, 0);
    vector<Mat1d> glcms;
    for (int i=0; i < 4; ++i)
    {
        glcms.emplace_back(nGrayBins, nGrayBins, 0);
    }

    int const d = 1;
    Rect middlePart(d, d, img.cols-2*d, img.rows-2*d);
    vector<Vec2i> pixelDisplacements = {{0, -d}, {d, -d}, {d, 0}, {d, d}};

    for (Point p : cvx::points(middlePart))
    {
        if (mask(p) != 0)
        {
            auto thisLevel = quantizedImage(p);

            for (int iAngle = 0; iAngle < 4; ++iAngle)
            {
                Point otherPoint = p + pixelDisplacements[iAngle];
                if (mask(otherPoint) != 0)
                {
                    auto otherLevel = quantizedImage(otherPoint);
                    ++glcms[iAngle](thisLevel, otherLevel);
                    ++nPairs[iAngle];
                }
            }
        }
    }

    vector<double> homogeneity(4,0);
    vector<double> energy(4,0);
    vector<double> entropy(4,0);

    double eps = std::numeric_limits<double>::epsilon();

    for (int iAngle : cvx::irange(4))
    {
        if (nPairs[iAngle] == 0)
        {
            continue;
        }

        glcms[iAngle] /= nPairs[iAngle];

        for (Point p : cvx::points({nGrayBins, nGrayBins}))
        {
			double freq = glcms[iAngle](p);
			homogeneity[iAngle] += freq / (1.0+cvx::sq(p.x-p.y));

			if (freq > eps)
			{
				entropy[iAngle] += -freq * std::log(freq);
			}
        }
        energy[iAngle] = cvx::FrobeniusSq(glcms[iAngle]);
    }

    return cvx::vectors::flatten(vector<vector<double>>{homogeneity, entropy, energy});
}
