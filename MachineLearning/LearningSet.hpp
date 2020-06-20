#ifndef LEARNINGSET_H
#define LEARNINGSET_H

#include <cvextra/coords.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/core/mat.hpp>
#include <vector>
#include <functional>

namespace crowd {

class LearningSet {

public:
    LearningSet(int inputDimensionality=0, int outputDimensionality=0);
    LearningSet(cv::InputArray regressionInputs, cv::InputArray groundTruths);

    void verticalAdd(cv::InputArray regressionInput, cv::InputArray groundTruth);
    void verticalAdd(LearningSet const& other);

    void horizontalAdd(cv::InputArray regressionInput, cv::InputArray groundTruth);
    void horizontalAdd(LearningSet const& other);

    auto verticalRange(cv::Range range) const -> LearningSet;
    auto verticalRange(int from, int to=cvx::END) const -> LearningSet;

    auto size() const -> double {return X.rows;}
    auto empty() const -> bool {return X.empty();}

    static
	auto verticalMerge(std::vector<LearningSet> const& sets) -> LearningSet;
    static
	auto horizontalMerge(std::vector<LearningSet> const& sets) -> LearningSet;

    cv::Mat1d X;
    cv::Mat1d Y;
};

class LearningSetup{
	LearningSet trainingSet;
	LearningSet testSet;
};

}

#endif // LEARNINGSET_H
