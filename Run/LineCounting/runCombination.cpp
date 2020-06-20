#include <Run/LineCounting/runCombination.hpp>
#include <Illustrate/fullIllustrate.hpp>
#include <Run/config.hpp>
#include <Run/LineCounting/scenarios.hpp>
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

#include <MachineLearning/NormalizedRegressionWithConfidence.hpp>
#include <MachineLearning/Regression.hpp>
#include <MachineLearning/Ridge.hpp>
#include <MachineLearning/NIGP.hpp>
#include <MachineLearning/KernelRidge.hpp>
#include <MachineLearning/HyperparameterOptimization/Hyperopt.hpp>
#include <Persistence.hpp>
#include <Python/Pyplot.hpp>

#include <cvextra/cvret.hpp>
#include <cvextra/io.hpp>
#include <cvextra/LoopRange.hpp>
#include <cvextra/mats.hpp>
#include <cvextra/vectors.hpp>
#include <boost/property_tree/json_parser.hpp>
#include <CrowdCounting/LineCounting/LineCounter.hpp>
#include <CrowdCounting/RegionCounting/AllToAllRegionCounter.hpp>
#include <CrowdCounting/RegionCounting/SharedModelRegionCounter.hpp>

#include <stdx/stdx.hpp>
#include <opencv2/opencv.hpp>
#include <algorithm>
#include <cmath>
#include <set>
#include <map>
#include <string>
#include <utility>
#include <vector>

using namespace std;
using namespace cv;
using namespace cvx;
using namespace crowd;
using namespace crowd::linecounting;


void CountingOutset::
loadOrCreate(string const& name, LineCountingScenario const& scenario, bool forceCreation)
{
//	if (!forceCreation && Persistence::canLoad(name+"predictedLineFlow"))
//	{
//		predictedLineFlow = Persistence::loadMat(name+"predictedLineFlow");
//		desiredLineFlow = Persistence::loadMat(name+"desiredLineFlow");
//		regionPredictedRegionCounts = Persistence::loadMat(name+"regionPredictedRegionCounts");
//		desiredRegionCounts = Persistence::loadMat(name+"desiredRegionCounts");
//		predictedLineFlowVariance = Persistence::loadMat(name+"predictedLineFlowVariance");
//		regionPredictionVariance = Persistence::loadMat(name+"regionPredictionVariance");
//	} else {
//		*this = createCombinationOutset(scenario);
//		Persistence::saveMat(name+"predictedLineFlow", predictedLineFlow);
//		Persistence::saveMat(name+"desiredLineFlow", desiredLineFlow);
//		Persistence::saveMat(name+"regionPredictedRegionCounts", regionPredictedRegionCounts);
//		Persistence::saveMat(name+"desiredRegionCounts", desiredRegionCounts);
//		Persistence::saveMat(name+"predictedLineFlowVariance", predictedLineFlowVariance);
//		Persistence::saveMat(name+"regionPredictionVariance", regionPredictionVariance);
//	}
}

void optimizeCombination(CountingOutset const& outset)
{
	Py::Object best =
		hyperopt::fmin(R"""(
		{
			'regionPenaltyFactor': hp.lognormal('regionPenaltyFactor', math.log(0.008), math.log(0.5/0.008)),
			'lineFlowPenaltyFactor': hp.lognormal('lineFlowPenaltyFactor', math.log(0.08), math.log(0.5/0.008)),
			'SORomega': hp.uniform('SORomega', 0.3, 1.9),
		})""",
		[&](Py::Tuple const& posi, Py::Dict const& keyw) -> Py::Object
		{
			cout << "trying: " << posi[0].str() << endl;

			pyx::EasyObject dict{posi[0]};
			double regionPenaltyFactor = dict["regionPenaltyFactor"];
			double lineFlowPenaltyFactor = dict["lineFlowPenaltyFactor"];
			double SORomega = dict["SORomega"];
			int SORiterations = 1000;//dict["SORiterations"];
			auto improved =
					crowd::linecounting::solveCombinationCoarseToFine(
							{outset.regionPredictedRegionCounts, outset.predictedLineFlow},
							{outset.desiredRegionCounts, outset.desiredLineFlow},
							lineFlowPenaltyFactor*cvret::divide(1.0, outset.predictedLineFlowVariance),
							2, 0.001,
							regionPenaltyFactor*cvret::divide(1.0, outset.regionPredictionVariance),
							0.000, SORomega, SORiterations);

			double loss = cvx::FrobeniusSq(improved.l-outset.desiredLineFlow);
			cout << "loss: " << loss << endl;
			return Py::Float{loss};
		},
		"algo=tpe.suggest, max_evals=8640, rstate=numpy.random.RandomState(1234578)");

	cout <<  "\n\nbest:\n " << best.str() << endl;
}

void crowd::run::
testCombination()
{
    auto lineAndRegion =
            LineAndRegionCombiner{

                SharedModelOverallLineCounter{
                    LineCounter{
                        NormalizedRegressionWithConfidence{
                            NIGP{
                                RegularizedLeastSquaresParam{
                                    1e10,
                                    0.1},
                                0.0201,
                                0.004412,
                                false}},
                        LineLearningRepresenter{
                            LineMultiFeatureExtractor{
                                HorizontalFlow{},
                                Foreground{},
                                //TextonHistogram{}
                            },
                            {20,5},
                            6, 2, 1, 1, 0.5, 1e-5, 0},
                        true},
                    7},

                SharedModelRegionCounter{
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
                    },
                Size{960,540}/4,
                0.008, 0.08, 0.65, 2500
    };

    auto scenario = crowd::getCrangeUltimateLineTestScenario();
    lineAndRegion.train(scenario.trainings);
    auto result = lineAndRegion.predictWithoutCombination(scenario.tests[0]);
    result.save(config::RESULT_PATH/"crange_ultimate");

    //auto result = FullResult::load(config::RESULT_PATH/"crange_ultimate");

    auto improved = lineAndRegion.combine(result);
    improved.plot(true, false).saveAndClose("/work/sarandi/crowd/crange_ultimate.png");
}
