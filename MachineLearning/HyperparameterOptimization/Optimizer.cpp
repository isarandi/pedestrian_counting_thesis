#include "Optimizer.hpp"

#include <cvextra/strings.hpp>
#include <cvextra/LoopRange.hpp>

using namespace std;
using namespace cv;
using namespace crowd;

double Optimizer::optimize()
{
    paramValues = vector<double>();
    for (auto param : parameters)
    {
        paramValues.push_back(param.initialValue);
    }

    optimize(parameters.size()-1);

    return currentError;
}

void Optimizer::evaluateCurrent()
{
    for (auto previousResult : previousResults)
    {
        if (previousResult.areParamsAlmostEqual(paramValues))
        {
            currentError = previousResult.evaluationResult;
            return;
        }
    }

    cout << "Trying parameters: " << cvx::str::join(paramValues, " | ") << "..." << endl;
    currentError = experiment->evaluate(paramValues);
    cout << "Done! Error: " << currentError << endl;

    previousResults.push_back(ExperimentResult{paramValues, currentError});
}

void Optimizer::optimize(int iParameter)
{
    if (iParameter == -1)
    {
        evaluateCurrent();
    }
    else
    {
        optimize(iParameter-1);

        //bool hasImprovedAtLeastOnce = false;
        double stepFactor = parameters[iParameter].stepFactor;
        for (int i : cvx::irange(4))
        {
            bool improved = optimize(iParameter, stepFactor);


            if (improved)
            {
                // zoom in on the part where we improved
                stepFactor = pow(stepFactor,1/4.0);
                cout << "Decrease scale" << endl;
                //       hasImprovedAtLeastOnce = true;
            }
            else
            {
                // increase scale, reason for no improvement may be
                // that we are moving on too small scale
                cout << "Increase scale" << endl;
                stepFactor = pow(stepFactor,2.0);
            }
        }
    }
}


bool Optimizer::optimize(int iParameter, double stepFactor)
{
    bool improvedByDecrease = optimizeInOneDirection(iParameter, 1.0/stepFactor);
    bool improvedByIncrease = optimizeInOneDirection(iParameter, stepFactor);

    return improvedByDecrease || improvedByIncrease;
}

bool Optimizer::optimizeInOneDirection(int iParameter, double stepFactor)
{
    bool improvedSubstantially = false;

    vector<double> bestParamsTillNow = paramValues;
    double bestErrorTillNow = currentError;
    int nStepsSinceLastImprovement = 0;

    while ((currentError < bestErrorTillNow*1.01) && (nStepsSinceLastImprovement<2))
    {
        paramValues[iParameter] *= stepFactor;
        optimize(iParameter-1);

        if (currentError < bestErrorTillNow)
        {
            stepFactor *= 1.3;

            if (currentError > bestErrorTillNow*0.9999)
            {
                // didn't improve substantially, try larger step
                stepFactor *= 2.0;
            } else {
                stepFactor *= 1.3;
                improvedSubstantially = true;
            }

            bestParamsTillNow = paramValues;
            bestErrorTillNow = currentError;
            nStepsSinceLastImprovement = 0;
        } else
        {
            ++nStepsSinceLastImprovement;
        }


    }

    paramValues = bestParamsTillNow;
    currentError = bestErrorTillNow;

    return improvedSubstantially;
}

bool ExperimentResult::areParamsAlmostEqual(std::vector<double> const& otherParamValues)
{
    for (int iParam : cvx::irange(paramValues.size()))
    {
        bool isAlmostEqual;
        if (paramValues[iParam] == 0)
        {
            isAlmostEqual = (otherParamValues[iParam]<1e-50);
        } else
        {
            double absRelDiff =
                    std::abs((otherParamValues[iParam]-paramValues[iParam])/paramValues[iParam]);
            isAlmostEqual = (absRelDiff < 1e-9);
        }

        if (!isAlmostEqual)
        {
            return false;
        }
    }
    return true;

}
