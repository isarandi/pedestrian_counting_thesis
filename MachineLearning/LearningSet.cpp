#include <MachineLearning/LearningSet.hpp>
#include <cvextra/coords.hpp>
#include <cvextra/mats.hpp>
#include <Persistence.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/core/mat.hpp>
#include <opencv2/core/core.hpp>
#include <vector>

using namespace std;
using namespace cv;
using namespace crowd;


LearningSet::
LearningSet(int inputDimensionality, int outputDimensionality)
    : X{0,inputDimensionality}
    , Y{0,outputDimensionality}
{}

LearningSet::
LearningSet(InputArray regressionInputs, InputArray groundTruths)
    : X{regressionInputs.getMat().clone()}
    , Y{groundTruths.getMat().clone()}
{}

void LearningSet::
verticalAdd(InputArray regressionInput, InputArray groundTruth)
{
    verticalAdd({regressionInput, groundTruth});
}

auto LearningSet::
verticalMerge(vector<LearningSet> const& sets) -> LearningSet
{
    LearningSet result{Mat1d{0, sets[0].X.cols}, Mat1d{0, sets[0].X.cols}};

    for (auto& set : sets)
    {
        cvx::vconcat(result.X, set.X, result.X);
        cvx::vconcat(result.Y, set.Y, result.Y);
    }

    return result;
}

auto LearningSet::
horizontalMerge(vector<LearningSet> const& sets) -> LearningSet
{
    LearningSet result{Mat1d{sets[0].X.rows, 0}, Mat1d{sets[0].X.rows, 0}};

    for (auto& set : sets)
    {
        cvx::hconcat(result.X, set.X, result.X);
        cvx::hconcat(result.Y, set.Y, result.Y);
    }

    return result;
}

auto LearningSet::
verticalRange(cv::Range range) const -> LearningSet
{
	return {
		X(range, Range::all()),
		Y(range, Range::all())};
}

auto LearningSet::
verticalRange(int from, int to) const -> LearningSet
{
	if (to == cvx::END)
	{
		to = X.rows;
	}

	return verticalRange({from,to});
}

void LearningSet::
verticalAdd(LearningSet const& other)
{
    if (other.empty())
    {
        return;
    } else if (this->empty())
    {
        X = other.X;
        Y = other.Y;
    } else {
        cvx::vconcat(this->X, other.X, this->X);
        cvx::vconcat(this->Y, other.Y, this->Y);
    }
}

void LearningSet::
horizontalAdd(
        cv::InputArray regressionInput,
        cv::InputArray groundTruth)
{
    horizontalAdd({regressionInput, groundTruth});
}

void LearningSet::
horizontalAdd(LearningSet const& other)
{
    if (other.empty())
    {
        return;
    } else if (this->empty())
    {
        X = other.X;
        Y = other.Y;
    } else {
        cvx::hconcat(this->X, other.X, this->X);
        cvx::hconcat(this->Y, other.Y, this->Y);
    }
}
