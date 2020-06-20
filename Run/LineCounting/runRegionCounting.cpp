#include "runRegionCounting.hpp"
#include <CrowdCounting/PersonLocations.hpp>
#include <CrowdCounting/Combination/Combiner.hpp>
#include <CrowdCounting/LineCounting/Features/LineFeatureExtractors.hpp>
#include <CrowdCounting/LineCounting/WindowBasedCounting.hpp>
#include <CrowdCounting/LineCounting/SlidingWindow.hpp>
#include <CrowdCounting/LineCounting/LineCountingExperiment.hpp>
#include <CrowdCounting/LineCounting/Features/Textons/TextonCreator.hpp>

#include <CrowdCounting/OverallLineCounting/SharedModelOverallLineCounter.hpp>
#include <CrowdCounting/OverallLineCounting/LineAndRegionCombiner.hpp>

#include <CrowdCounting/RegionCounting/CountingFrame.hpp>
#include <CrowdCounting/RegionCounting/CountingTestCase.hpp>
#include <CrowdCounting/RegionCounting/CountingTestResult.hpp>
#include <CrowdCounting/RegionCounting/Features/FeatureExtractors.hpp>
#include <CrowdCounting/RegionCounting/Features/MultiFeatureExtractor.hpp>
#include <CrowdCounting/RegionCounting/FrameCollection.hpp>
#include <CrowdCounting/RegionCounting/SharedModelRegionCounter.hpp>
#include <CrowdCounting/RegionCounting/AllToAllRegionCounter.hpp>
#include <CrowdCounting/PersonLocations.hpp>

#include <MachineLearning/HyperparameterOptimization/Hyperopt.hpp>
#include <MachineLearning/KernelRidge.hpp>
#include <MachineLearning/NormalizedRegressionWithConfidence.hpp>
#include <MachineLearning/Ridge.hpp>
#include <MachineLearning/NIGP.hpp>

#include <Run/config.hpp>
#include <Run/LineCounting/scenarios.hpp>
#include <Run/LineCounting/runCombination.hpp>

#include <Python/EasyObject.hpp>
#include <Python/Pyplot.hpp>
#include <Python/PythonEnvironment.hpp>

#include <Persistence.hpp>

#include <Illustrate/fullIllustrate.hpp>

#include <cvextra/colors.hpp>
#include <cvextra/cvenums.hpp>
#include <cvextra/cvret.hpp>
#include <cvextra/gui.hpp>
#include <cvextra/LoopRange.hpp>
#include <cvextra/math.hpp>
#include <cvextra/strings.hpp>
#include <cvextra/visualize.hpp>
#include <cvextra/io.hpp>
#include <cvextra/vectors.hpp>
#include <cvextra/ImageLoadingIterable.hpp>
#include <CrowdCounting/RegionCounting/AllToAllRegionCounter.hpp>

#include <opencv2/core/core.hpp>
#include <opencv2/core/mat.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <algorithm>
#include <cmath>
#include <iostream>
#include <map>
#include <set>
#include <string>
#include <vector>

using namespace std;
using namespace cv;
using namespace cvx;
using namespace crowd;
using namespace crowd::linecounting;

using namespace crowd::run;

//vidd
//{"useOtherSegmentsForTraining", 0, 2, 0},
//{"leftRightIntertwine", 0, 2, 2},
//{"dirLeft", 0, 2, 0},
//{"normalizeRegression", 0, 2, 2},
//{"augmentWithMirrored", 0, 2, 2},
//{"logKernelRidgeC", -5, 10, 10},
//{"logKernelRidgeGamma", -10, 15, 2.062},
//{"nMicroSections", 1, 10, 3},
//{"sectionStepInMicroSections", 1, 10, 1},
//{"sectionSizeInMicroSections", 1, 10, 2},
//{"windowSize", 10, 200, 20},
//{"stepSize", 1, 200, 5},
//{"trainingWeightOfInvisiblePeople", 0, 1, 0.8},
//{"signalVarianceMultiplier", 0.001, 0.5, 0.13},


static
void tweakLearning()
{
    LineCountingScenario scenario = getDensitiesToLongScenario();

//	cvx::gui::startTweaking({
//        {"flowSegmentationThresh", 0, 5, 1},
//    },
//    [&](map<string,double> const& params){
//
//    });
}

void crowd::run::
testRegionCounting()
{
    hyperopt::fminSimple(R"""(
    {
        'logC': hp.normal('logC', 13, 10),
        #'logGamma': hp.normal('logGamma', -5.4, 3),
        #'logInputVar': hp.uniform('logInputVar', -4.7, -4.3),
    })""",
    [&](pyx::EasyObject p)
    {
        auto regionCounter =
                AllToAllRegionCounter{
                    Size{},
                    MultiFeatureExtractor{
                        Area{},
                        Perimeter{},
                        PerimeterToAreaRatio{},
                        Edges{},
                        EdgeOriHistogram{6},
                        PerimeterOriHistogram{6},
                        GrayLevelCooccurrence{},
                        EdgeMinkowski{15},
//                        StatisticalLandscape(31),
//                        FilterBank::LM(49),
//                        LocalBinaryPatterns(1,8,15,Size(1,1),true),
                    },
                    NormalizedRegressionWithConfidence{
                        Ridge{std::exp(-5.1), 1e-3}},
//                        NIGP{
//                            RegularizedLeastSquaresParam{
//                                std::exp(-4.5),//std::exp((double)p["logC"]),
//                                0.13},
//                            std::exp(-5.07),//std::exp((double)p["logGamma"]),
//                            std::exp(-4.5),//std::exp((double)p["logInputVar"]),
//                            true}},
                    };

        auto fullResults =
                crowd::getViddTestScenario()
                    .evaluateRegion(regionCounter, 7, Size{0,0});//, Size{960,540}/4);

        for (auto const& res : fullResults)
        {
            cout << res.meanRegionAbsRelError(res.predictedRegionCounts.mean) << endl;
        }

        auto aggregate = FullResult::horizontalMerge(fullResults);

        fullResults[0].regionPlot(false, false).saveAndClose("/work/sarandi/crowd/crange_region_test_ridge_alltoall_0.png");
        //fullResults[1].regionPlot(false).saveAndClose("/work/sarandi/crowd/region_valid_ridge_1.png");
        //fullResults[2].regionPlot(false).saveAndClose("/work/sarandi/crowd/region_valid_ridge_2.png");

        aggregate.predictedRegionCounts.mean = cvxret::medianFilter(aggregate.predictedRegionCounts.mean, Size{1,5});
        cv::GaussianBlur(aggregate.predictedRegionCounts.mean, aggregate.predictedRegionCounts.mean, Size{1,9}, 2.0,3.0, BORDER_REPLICATE);

        cout << aggregate.meanRegionAbsError(aggregate.predictedRegionCounts.mean) << endl;
        cout << aggregate.meanRegionAbsRelError(aggregate.predictedRegionCounts.mean) << endl;

        return aggregate.meanRegionAbsError(aggregate.predictedRegionCounts.mean);
    },
    "algo=tpe.suggest, max_evals=100000, rstate=numpy.random.RandomState(1234567)");
}



