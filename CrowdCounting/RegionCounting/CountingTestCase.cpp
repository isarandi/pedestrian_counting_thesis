#include <cvextra/strings.hpp>
#include <CrowdCounting/RegionCounting/CountingTestCase.hpp>
#include <CrowdCounting/RegionCounting/CountingTestResult.hpp>
#include <opencv2/core/core.hpp>
#include <iostream>
#include <sstream>

using namespace std;
using namespace stdx;
using namespace cv;
using namespace cvx;
using namespace crowd;

CountingTestResult CountingTestCase::run()
{
    cout << getDescription() << endl;

    counter->train(training);
    PredictionWithConfidence predictions = counter->predictWithConfidence(test);

    return CountingTestResult(*this, predictions);
}

string CountingTestCase::getDescription() const
{
    stringstream ss;
    ss
            << "[" << cvx::timestamp() << "] - " << name << endl
            << "  Training: " << training.getDescription() << endl
            << "  Test: " << test.getDescription() << endl
            << "  Counting model:" << endl
            << cvx::str::indentBlock(counter->getDescription(), 2);

    return ss.str();
}
