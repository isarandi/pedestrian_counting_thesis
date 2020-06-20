#include <MachineLearning/Regression.hpp>
#include <MachineLearning/KernelRidge.hpp>
#include <MachineLearning/NIGP.hpp>
#include <MachineLearning/Ridge.hpp>
#include <MachineLearning/NormalizedRegressionWithConfidence.hpp>

using namespace crowd;

auto Regression::
create(
		boost::property_tree::ptree const& pt
		) -> std::unique_ptr<Regression>
{
	std::string type = pt.get<std::string>("type");
	if (type == "KernelRidge")
	{
		return KernelRidge::create(pt);
	} else if (type == "Ridge") {
		return Ridge::create(pt);
	} else if (type == "NIGP") {
		return NIGP::create(pt);
	} else if (type == "NormalizedRegressionWithConfidence") {
		return NormalizedRegressionWithConfidence::create(pt);
	}
	throw 1;
}

auto RegressionWithConfidence::
create(
		boost::property_tree::ptree const& pt
		) -> std::unique_ptr<RegressionWithConfidence>
{
	std::string type = pt.get<std::string>("type");
	if (type == "KernelRidge")
	{
		return KernelRidge::create(pt);
	} else if (type == "NIGP") {
		return NIGP::create(pt);
	} else if (type == "NormalizedRegressionWithConfidence") {
		return NormalizedRegressionWithConfidence::create(pt);
	}
	throw 1;
}
