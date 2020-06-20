#include "runLineCounting.hpp"
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

void tweakLearning()
{
	cvx::gui::startTweaking({
        {"augmentWithMirrored", 0, 2, 2},
        {"logKernelRidgeC", -5, 10, 9},
        {"logKernelRidgeGamma", -10, 15, -2.38},
        {"nMicroSections", 1, 10, 6},
        {"sectionStepInMicroSections", 1, 10, 1},
        {"sectionSizeInMicroSections", 1, 10, 2},
        {"segmentationThreshold", 0.01, 10, 1.0},
        {"windowSize", 10, 200, 20},
        {"stepSize", 1, 200, 5},
        {"signalVarianceMultiplier", 0.001, 0.5, 0.13},
    },
    [&](map<string,double> const& p){
        auto lineCounter =
                LineCounter{
                    NormalizedRegressionWithConfidence{
                        NIGP{
                            RegularizedLeastSquaresParam{
                                1e10,
                                0.1},
                                0.0201,0.004412,
                                true}},
                    LineLearningRepresenter{
                        LineMultiFeatureExtractor{
                            HorizontalFlow{},
                            Foreground{},
                            //CannySlice{},
                            //TextonHistogram{}
                        },
                        {20,5},
                        6, 2, 1, 1, p.at("segmentationThreshold"), 1e-5, 0},
                    true};

        auto fullResults =
                crowd::getViddValidationScenario()
                    .evaluate(SharedModelOverallLineCounter{lineCounter, 7});

        auto aggregate = FullResult::horizontalMerge(fullResults);

        return fullResults[0].linePlot(true, false).renderAndClose();
    }, true);
}

void listResults(){

    auto paths = cvx::filesystem::listdir(config::RESULT_PATH, "[^\\.]+");

    Mat1d precisions{(int)paths.size(), 1};
    Mat1d recalls{(int)paths.size(), 1};

    int iPath = 0;
    for (auto path : paths)
    {
        cout << path << endl;
        auto desc = cvx::io::readFile(path);
        if (desc.find("Cov") == std::string::npos)
        {
            continue;
        }

        Confusion conf{};
        for (int i : cvx::irange(3))
        {
            auto result = FullResult::load(path.string()+std::to_string(i));
            conf += result.confusion(result.predictedLineFlow.mean, 37);
        }

        precisions(iPath) = conf.precision();
        recalls(iPath) = conf.recall();

        cout << desc << conf.precision() << " " << conf.recall() << " " << conf.f1() << endl << endl;
        ++iPath;
    }

    pyx::Pyplot plt;
    plt.scatter(precisions, recalls);
    plt.xlim(0,1);
    plt.ylim(0,1);
    plt.showAndClose();
}

void crowd::run::
testLineCounting()
{

    //listResults();
	//tweakLearning();
//    'windowSize': hp.qnormal('windowSize', 20, 5, 1),
//    'windowStepFactor': hp.normal('windowStepFactor', 0.25, 0.08),
//    'nMicroSections': hp.qnormal('nMicroSections', 6, 2, 1),
//    'nOutputSections': hp.choice('nOutputSections', [1, 2, 3]),

    hyperopt::fminSimple(R"""(
    {
        'logC': hp.normal('logC', 20, 8),
        'logGamma': hp.normal('logGamma', -4.5, 2),
        'logInputVar': hp.uniform('logInputVar', -6, -2),
    })""",
    [&](pyx::EasyObject p)
    {



        auto lineCounter =
                LineCounter{
                    NormalizedRegressionWithConfidence{
                        //Ridge{1e-3,1e-2}},
                        NIGP{
                            RegularizedLeastSquaresParam{
                                1e10,
                                0.1},
                                0.0201,
                                0.004412, true, 1}},
                    LineLearningRepresenter{
                        LineMultiFeatureExtractor{
                            HorizontalFlow{},
                            Foreground{},
                            //TextonHistogram{}
                        },
                        {20,5},
                        6, 2, 1, 1, 1.0, 1e-5, 0},
                    true};

        auto scenario = crowd::getSmallCrangeScenario();

        crowd::fullIllustrate(scenario, SharedModelOverallLineCounter{lineCounter, 7}, config::DATA_PATH/"crange_long.avi",1);

        auto loc = config::locations(scenario.tests[0].datasetName).betweenFrames(scenario.tests[0].frameRange);
        auto flows = loc.getInstantFlow({{480,0},{480,540}}, 1e-5, 0);
        pyx::Pyplot plt;
        plt.plot(cvxret::cumsum(flows.col(0)));
        plt.plot(cvxret::cumsum(flows.col(1)));
        plt.showAndClose();

        auto fullResults = scenario.evaluate(SharedModelOverallLineCounter{lineCounter, 7});

        //crowd::fullIllustrate(scenario, fullResults[0], config::DATA_PATH/"crange_long_texton.avi", 1.0);
        fullResults[0].plot(true,true);

        auto aggregate = FullResult::horizontalMerge(fullResults);
        aggregate.save(config::RESULT_PATH/"ultimate");
////
//        fullResults[0].linePlot(true, false).saveAndClose("/work/sarandi/crowd/test_crange_normal_0.png");
//        //fullResults[0].regionFromLinePlot(false).saveAndClose("/work/sarandi/crowd/test_crange_normal_nomirror_line0.png");
//        auto precrec = fullResults[0].precisionRecallCurve(fullResults[0].predictedLineFlow.mean, 300);
//
//        pyx::Pyplot plt;
//        plt.plot(precrec.col(0), "r");
//        plt.plot(precrec.col(1), "g");
//        plt.plot(precrec.col(2), "b");
//        plt.saveAndClose("/work/sarandi/crowd/test_crange_normal_nomirror_prec_rec_line0.png");

//        fullResults[1].linePlot(true, false).saveAndClose("/work/sarandi/crowd/valid_ridge_1.png");
//        fullResults[2].linePlot(true, false).saveAndClose("/work/sarandi/crowd/valid_ridge_2.png");
//
//        fullResults[0].linePlot(false, false).saveAndClose("/work/sarandi/crowd/valid_ridge_0_inst.png");
//        fullResults[1].linePlot(false, false).saveAndClose("/work/sarandi/crowd/valid_ridge_1_inst.png");
//        fullResults[2].linePlot(false, false).saveAndClose("/work/sarandi/crowd/valid_ridge_2_inst.png");

//
//
//
//        for (auto const& fr : fullResults)
//        {
//            fr.plot(true, true);
//            fr.plot(false, true);
//
//            auto curve = fr.boxEvaluationCurve(fr.predictedLineFlow.mean, true);
//            pyx::Pyplot plt;
//            plt.plot(curve.reshape(1).col(0));
//            plt.plot(curve.reshape(1).col(1));
//            plt.saveAndClose("/work/sarandi/crowd/results/");
//        }

        auto conf = aggregate.confusion(aggregate.predictedLineFlow.mean, 37);
//
//        {
//            auto box = aggregate.boxEvaluationCurve(aggregate.predictedLineFlow.mean, true);
//
//            pyx::Pyplot plt;
//            plt.plot(box.reshape(1).col(0), "b--");
//            plt.plot(box.reshape(1).col(1), "k");
//            //plt.plot(ec.getExpectedAbsErrorCurve(aggregate.predictedLineFlow.mean.rows));
//            plt.saveAndClose("/work/sarandi/crowd/test_crange_normal_nomirror_boxcurve.png");
//        }
//
//        auto ec = aggregate.errorCharacteristics(aggregate.predictedLineFlow.mean, 50);
//        cout << ec.mean << " " << std::sqrt(ec.variance) << endl;
//
//        cout
//            << conf.precision() << " "
//            << conf.recall() << " "
//            << conf.f1() << " "
//            << aggregate.meanFinalAbsError(aggregate.predictedLineFlow.mean) << " "
//            << aggregate.meanFinalAbsRelError(aggregate.predictedLineFlow.mean) << endl;
//
        return 1-conf.f1();
    },
    "algo=tpe.suggest, max_evals=100000, rstate=numpy.random.RandomState(1234567)");

    //TextonCreator t{Size{960,540}/4, 5};
    //t.train(
}



