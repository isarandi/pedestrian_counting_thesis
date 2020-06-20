#include <boost/filesystem.hpp>

#include <CrowdCounting/RegionCounting/CountingFrame.hpp>
#include <CrowdCounting/RegionCounting/CountingTestCase.hpp>
#include <CrowdCounting/RegionCounting/CountingTestResult.hpp>
#include <CrowdCounting/CrowdCountingUtils.hpp>
#include <CrowdCounting/RegionCounting/FrameCollection.hpp>

#include <cvextra/cvret.hpp>
#include <cvextra/mats.hpp>
#include <cvextra/strings.hpp>
#include <cvextra/utils.hpp>
#include <cvextra/LoopRange.hpp>
#include <cvextra/io.hpp>
#include <cvextra/filesystem.hpp>
#include <CrowdCounting/RegionCounting/RegionCounter.hpp>

#include <opencv2/core/core_c.h>
#include <opencv2/core/mat.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <stdx/stdx.hpp>
#include <cmath>
#include <fstream>
#include <sstream>
#include <vector>

using namespace std;
using namespace stdx;
using namespace cv;
using namespace cvx;
using namespace crowd;


CountingTestResult::
CountingTestResult(
        CountingTestCase const& testCase,
		PredictionWithConfidence const& predictions)
    : testCaseClone(testCase.clone())
    , predictions(predictions)
{
    desired = cvx::mats::matFromRows(
                crowd::gridQuantizeAll(
                    testCaseClone->getTestSet().getFrames(),
                    testCaseClone->getCrowdCounter().getGridSize()));

    calculateErrorMeasures();
}

void CountingTestResult::
calculateErrorMeasures()
{
    double sumOfSquaredErrors = 0;
    double sumOfAbsoluteErrors = 0;
    double sumOfRelativeErrors = 0;

    int nFrames = predictions.mean.rows;
    for (int iFrame = 0; iFrame < nFrames; ++iFrame)
    {
        double desiredSum = 0;
        double predictedSum = 0;

        for (int iCell = 0; iCell < predictions.mean.cols; ++iCell)
        {
            predictedSum += predictions.mean(iFrame,iCell);
            desiredSum += desired(iFrame,iCell);
        }

        double diff = predictedSum - desiredSum;

        sumOfSquaredErrors += diff*diff;
        sumOfAbsoluteErrors += std::abs(diff);
        sumOfRelativeErrors += std::abs(diff/desiredSum);
    }

    mse = sumOfSquaredErrors /nFrames;
    mae = sumOfAbsoluteErrors/nFrames;
    mre = sumOfRelativeErrors/nFrames;
}

auto CountingTestResult::
getResultDescription() const -> string
{
    return cvx::str::format(
            "MSE = %.4f, MAE = %.4f, MRE = %.4f",
            getMeanSquaredError(),
            getMeanAbsoluteError(),
            getMeanRelativeError());
}

void CountingTestResult::
saveToLog() const
{
    ofstream logFile;
    logFile.open("/work/sarandi/crowd/results/experiment_log.txt",
                 std::ios_base::app);
    logFile << testCaseClone->getDescription() << endl;
    logFile << getResultDescription() << endl << endl;
}

void CountingTestResult::
saveImages() const
{
    Size gridSize = testCaseClone->getCrowdCounter().getGridSize();
    auto countingFrames = testCaseClone->getTestSet().getFrames();

    bpath illustFolderPath =
            "/work/sarandi/crowd/results/count_illustrations/counts_"
            + cvx::timestamp();

    for (int iFrame : cvx::irange(countingFrames.size()))
    {
        Mat illust =
                crowd::generateCountingIllustration(
                    countingFrames[iFrame].getFullResolutionFrame(),
                    desired.row(iFrame),
                    predictions.mean.row(iFrame),
                    gridSize);

        cvx::imwrite(
                illustFolderPath/cvx::str::format("frame_%06d.png", iFrame),
                illust);
    }
}

void CountingTestResult::
saveMatlabCode() const
{
    string folderPath = "/work/sarandi/crowd/results/matlab_scripts/";
    boost::filesystem::create_directories(folderPath);

    string time = cvx::timestamp();
    string name = testCaseClone->getName();

    ofstream matlabScript(cvx::str::format("%s/m_%s_%s.m", folderPath, time, name));

    matlabScript << cvx::str::prependToLines(testCaseClone->getDescription(), "% ") << endl;
    matlabScript << "% " << getResultDescription() << endl << endl;

    matlabScript << "pred_" << name << " = " << predictions.mean << ";" << endl;
    matlabScript << "gt = " << desired << ";" << endl;

    matlabScript << "sum_pred_" << name << " = sum(pred_" << name << ", 2);" << endl;
    matlabScript << "sum_gt = sum(gt, 2);" << endl;
}

void CountingTestResult::
savePlot() const
{
    string folderPath = "/work/sarandi/crowd/results/plots/";
    boost::filesystem::create_directories(folderPath);

    string time = cvx::timestamp();
    string name = testCaseClone->getName();

    string descriptionFilePath = cvx::str::format("%s/m_%s_%s.txt", folderPath, time, name);

    ofstream descriptionFile(descriptionFilePath);
    descriptionFile << testCaseClone->getDescription() << endl << endl;
    descriptionFile << getResultDescription();
    descriptionFile.close();

//    auto fig = figure();

//    string matlabFilePath = cvx::str::format("%s/m_%s_%s.m", folderPath, time, name);
//    ofstream matlabFile(matlabFilePath);
//    matlabFile << fig.matlabScript();
//    matlabFile.close();

//    string plotFilePath = cvx::str::format("%s/m_%s_%s.png", folderPath, time, name);
//    cv::imwrite(plotFilePath, fig.render(Size(1500,300)));
}


