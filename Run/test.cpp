#include <CrowdCounting/RegionCounting/CountingFrame.hpp>
#include <CrowdCounting/RegionCounting/CountingTestCase.hpp>
#include <CrowdCounting/RegionCounting/CountingTestResult.hpp>
#include <CrowdCounting/RegionCounting/Features/Extractors/Area.hpp>
#include <CrowdCounting/RegionCounting/Features/Extractors/LocalBinaryPatterns.hpp>
#include <CrowdCounting/RegionCounting/Features/MultiFeatureExtractor.hpp>
#include <CrowdCounting/RegionCounting/FrameCollection.hpp>
#include <CrowdCounting/OverallLineCounting/OverallLineCounter.hpp>
#include <CrowdCounting/PersonLocations.hpp>
#include <MachineLearning/KernelRidge.hpp>
#include <MachineLearning/Ridge.hpp>
#include <opencv2/opencv.hpp>
#include <cvextra.hpp>
#include <cvextra/LoopRange.hpp>
#include <CrowdCounting/RegionCounting/AllToAllRegionCounter.hpp>

#include <Run/config.hpp>
#include <cmath>
#include <vector>

using namespace std;
using namespace cv;
using namespace cvx;
using namespace crowd;
using namespace crowd::linecounting;

void test()
{

    vector<OverallLineCountingSet> datasets = OverallLineCountingSet::makeMany({
    		"ucsd_vidd",
    		"crange_ausschnitt1",
            "crange_densities/12",
            "crange_densities/13",
            "crange_densities/21",
            "crange_densities/22",
            "crange_densities/23",
            "crange_densities/24",
            "crange_densities/31",
            "crange_densities/32",
            "crange_densities/33",
            "crange_densities/34",
            "crange_densities/41",
            "crange_densities/42",
            "crange_densities/43",
            "crange_densities/44",
    });

//    for (int i=0; i<4500; i+=250)
//    {
//        datasets.push_back(OverallLineCountingSet{"crange_ausschnitt1", {i, i+250}});
//    }

    for (auto const& ds : datasets)
    {
    	Size frameSize = config::frames(ds.datasetName)[0].size();
    	LineSegment seg {{frameSize.width/2, 0},{frameSize.width/2,frameSize.height-1}};

    	PersonLocations loc =
    	        config::locations(ds.datasetName)
    	            .betweenFrames(ds.frameRange)
    	            .applyStencil(config::roiStencil(ds.datasetName));

    	auto byFrame = loc.getGroupedByFrame();
    	double meanCount =
    			cv::mean(
    					cvx::mats::matFromRows(
    							{cvx::vectors::transform(
    									byFrame,
										[](vector<PersonInstance> const& pis){return (double)pis.size();})}))[0];

    	auto instances = loc.getInstances();
    	double visibilityPercentage = std::count_if(instances.begin(), instances.end(), [](PersonInstance pi){return pi.visible;})/(double)instances.size();

    	int nIndividuals = loc.getGroupedByPerson().size();
    	Mat1d meanFlowPerSecond = 25*cvret::reduce(loc.getInstantFlow(seg, 10, 50), 0, REDUCE_AVG);

    	cout << cvx::str::format("%s; %d; %d; %.2f; %.2f; %.2f; %d; %.2f", ds.datasetName, ds.frameRange.start, ds.frameRange.end, meanCount, meanFlowPerSecond(0), meanFlowPerSecond(1), nIndividuals, 1-visibilityPercentage) << endl;
    }

//    Mat1b img = cvx::mats::matFromRows({{1,2,0},{0,0,0},{0,0,0}});
//    Mat labels;
//    int nLabels = cvx::connectedComponents(img, labels, 8, CV_8U);
//
//    std::cout<<nLabels<<endl;

}
