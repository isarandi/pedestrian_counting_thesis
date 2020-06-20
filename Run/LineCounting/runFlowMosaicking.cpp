#include <cvextra/ImageLoadingIterable.hpp>
#include <cvextra/LoopRange.hpp>
#include <cvextra/math.hpp>
#include <cvextra/io.hpp>
#include <CrowdCounting/LineCounting/FeatureSlices.hpp>
#include <CrowdCounting/LineCounting/LineCountingSet.hpp>
#include <CrowdCounting/OverallLineCounting/FullResult.hpp>
#include <CrowdCounting/LineCounting/FlowMosaicking/FlowMosaicCounter.hpp>
#include <CrowdCounting/PersonLocations.hpp>
#include <MachineLearning/Ridge.hpp>
#include <MachineLearning/NormalizedRegressionWithConfidence.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/core/mat.hpp>
#include <Run/config.hpp>
#include <Run/LineCounting/scenarios.hpp>
#include <Run/LineCounting/runFlowMosaicking.hpp>

#include <Python/EasyCallable.hpp>
#include <Python/EasyObject.hpp>
#include <Python/Pyplot.hpp>
#include <Python/PythonEnvironment.hpp>

#include <string>

using namespace crowd;
using namespace crowd::linecounting;
using namespace cv;
using namespace cvx;
using namespace std;

void crowd::linecounting::
runFlowMosaicking()
{
    auto mosaicCounter =
            FlowMosaicCounter{
                NormalizedRegressionWithConfidence{
                    Ridge{1e-3, 1e-5}},
//                    NIGP{
//                        RegularizedLeastSquaresParam{
//                            std::exp((double)p["logC"]),
//                            0.33},
//                            std::exp((double)p["logGamma"]),
//                        std::exp((double)p["logInputVar"]), true}},
                //7, 30, 1000, 1.0, false}; // VIDD
                1, 300, 15000, 1.0, true}; //CRANGE

    auto fullResults =
            crowd::getCrangeLineTestScenario()
                .evaluate(mosaicCounter);

    auto aggregate = FullResult::horizontalMerge(fullResults);
//
    fullResults[0].linePlot(true, false).saveAndClose("/work/sarandi/crowd/test_crange_flowmos_0.png");
    auto conf = aggregate.confusion(aggregate.predictedLineFlow.mean, 37);

    {
        auto box = aggregate.boxEvaluationCurve(aggregate.predictedLineFlow.mean, true);

        pyx::Pyplot plt;
        plt.plot(box.reshape(1).col(0), "b--");
        plt.plot(box.reshape(1).col(1), "k");
        //plt.plot(ec.getExpectedAbsErrorCurve(aggregate.predictedLineFlow.mean.rows));
        plt.saveAndClose("/work/sarandi/crowd/test_crange_flowmos_boxcurve.png");
    }

    auto ec = aggregate.errorCharacteristics(aggregate.predictedLineFlow.mean, 50);
    cout << ec.mean << " " << std::sqrt(ec.variance) << endl;

    cout
        << conf.precision() << " "
        << conf.recall() << " "
        << conf.f1() << " "
        << aggregate.meanFinalAbsError(aggregate.predictedLineFlow.mean) << " "
        << aggregate.meanFinalAbsRelError(aggregate.predictedLineFlow.mean) << endl;


}
