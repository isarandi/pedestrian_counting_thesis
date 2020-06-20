#ifndef STRINGUTILS_HPP
#define STRINGUTILS_HPP

#include <vector>
#include <string>
#include <sstream>

#include <boost/format.hpp>

#include <cstring>
#include <type_traits>

namespace cvx {

namespace str
{
    namespace details
    {
        auto formatImpl(boost::format& f) -> boost::format&;

        template <typename Head, typename... Tail>
        auto formatImpl(boost::format& f, Head const& head, Tail&&... tail) -> boost::format&
        {
            return formatImpl(f % head, tail...);
        }
    }

    template <typename T>
    auto join(std::vector<T> const& v, std::string const& separator) -> std::string
    {
        std::stringstream ss(std::stringstream::out);

        for (int i = 0; i < v.size()-1; ++i)
        {
            ss << v[i] << separator;
        }
        ss << v[v.size()-1];

        return ss.str();
    }

    auto join(std::vector<double> const& v, int nDecimalsAfterPoint, std::string const& separator) -> std::string;
    auto zeropad(int number, int desiredWidth) -> std::string;
    auto repeat(std::string const& str, int times) -> std::string;
    auto replace(std::string const& str, std::string const& oldString, std::string const& newString) -> std::string;
    auto split(std::string const& str, std::string const& splitter) -> std::vector<std::string>;
    auto prependToLines(std::string const& block, std::string const& prefix) -> std::string;
    auto indentBlock(std::string const& block, int nLevels=1) -> std::string;

    template <typename... Args>
    auto format(std::string const& formatString, Args&&... args) -> std::string
    {
        boost::format f(formatString);
        return details::formatImpl(f, args...).str();
    }

    auto toVariableName(std::string s) -> std::string;
}
}

#endif // STRINGUTILS_HPP
