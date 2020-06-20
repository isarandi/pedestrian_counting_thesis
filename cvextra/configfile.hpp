#ifndef CVEXTRA_CONFIGFILE_HPP_
#define CVEXTRA_CONFIGFILE_HPP_

#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/json_parser.hpp>
#include <memory>
#include <vector>

#define CVX_CONFIG_DERIVED(type) \
	virtual auto describe() const -> boost::property_tree::ptree; \
    static auto create(boost::property_tree::ptree const& pt) -> std::unique_ptr<type>;

#define CVX_CONFIG_BASE(type) \
	virtual auto describe() const -> boost::property_tree::ptree = 0; \
    static auto create(boost::property_tree::ptree const& pt) -> std::unique_ptr<type>;

#define CVX_CONFIG_SINGLE(type) \
	auto describe() const -> boost::property_tree::ptree; \
    static auto create(boost::property_tree::ptree const& pt) -> std::unique_ptr<type>;

namespace cvx {
namespace configfile {

inline
auto json_str(boost::property_tree::ptree const& pt) -> std::string
{
    std::stringstream ss;
    boost::property_tree::write_json(ss, pt, false);
    return ss.str();
}

template <typename T>
auto describeCollection(T const& collection) -> boost::property_tree::ptree
{
	boost::property_tree::ptree pt;
	for (auto const& elem : collection)
	{
		pt.add_child("item", elem.describe());
	}
	return pt;
}

template <typename T>
auto loadVectorPtr(boost::property_tree::ptree const& pt) -> std::vector<std::unique_ptr<T>>
{
	std::vector<std::unique_ptr<T>> items;
	for (auto const& elem : pt)
	{
		items.emplace_back(std::move(T::create(elem.second)));
	}
	return items;
}

template <typename T>
auto loadVector(boost::property_tree::ptree const& pt) -> std::vector<T>
{
	std::vector<T> items;
	for (auto const& elem : pt)
	{
		items.emplace_back(*T::create(elem.second));
	}
	return items;
}

}}

#endif
