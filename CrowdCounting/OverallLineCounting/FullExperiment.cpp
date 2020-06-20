//#include <CrowdCounting/OverallLineCounting/FullExperiment.hpp>
//
//using namespace std;
//using namespace cv;
//using namespace cvx;
//using namespace crowd;
//using namespace crowd::linecounting;
//
//void FullExperiment::
//run()
//{
//	combiner.train(trainings);
//	//result = stdx::make_unique<FullResult>(combiner.predictWithCombination(test));
//}
//
//auto FullExperiment::
//describe() const -> boost::property_tree::ptree
//{
//	boost::property_tree::ptree pt;
//	pt.add_child("trainings", cvx::configfile::describeCollection(trainings));
//	pt.add_child("tests", cvx::configfile::describeCollection(trainings));
//	pt.add_child("combiner", combiner.describe());
//
//	return pt;
//}
//
//auto FullExperiment::
//create(boost::property_tree::ptree const& pt) -> std::unique_ptr<FullExperiment>
//{
//	return stdx::make_unique<FullExperiment>(
//			cvx::configfile::loadVector<OverallLineCountingSet>(pt.get_child("trainings")),
//			*OverallLineCountingSet::create(pt.get_child("test")),
//			*LineAndRegionCombiner::create(pt.get_child("combiner")))
//	);
//}
