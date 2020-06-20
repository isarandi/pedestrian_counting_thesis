#include "strings.hpp"

#include <boost/algorithm/string.hpp>
#include <boost/regex.hpp>

#include <sstream>
#include <iomanip>

using namespace std;

string cvx::str::join(vector<double> const& v, int nDecimalsAfterPoint, string const& separator)
{
    if (v.empty()) {return "";}

    string formatString = cvx::str::format("%%f.%d", nDecimalsAfterPoint);

    stringstream ss;
    for (int i = 0; i < v.size()-1; ++i) {
        ss << cvx::str::format(formatString, v[i]) << separator;
    }
    ss << cvx::str::format(formatString, v[v.size()-1]);

    return ss.str();
}

string cvx::str::zeropad(int number, int desiredWidth)
{
    stringstream ss;
    ss << std::setw(desiredWidth) << std::setfill('0') << number;
    return ss.str();
}

string cvx::str::repeat(string const& str, int times)
{
    stringstream ss;
    for (int i = 0; i < times; ++i)
    {
        ss << str;
    }
    return ss.str();
}

vector<string> cvx::str::split(string const& str, string const& splitter)
{
    vector<string> parts;
    boost::algorithm::split(parts, str, boost::algorithm::is_any_of(splitter));
    return parts;
}

string cvx::str::prependToLines(string const& block, string const& prefix)
{
    return prefix +
            boost::algorithm::join(
                cvx::str::split(block, "\n"),
                "\n"+prefix);
}

auto cvx::str::
replace(
		std::string const& str,
		std::string const& oldString,
		std::string const& newString
		) -> std::string
{
	return boost::algorithm::join(
                cvx::str::split(str, oldString),
				newString);
}

string cvx::str::indentBlock(string const& block, int byLevel)
{
    string indentation = cvx::str::repeat("  ", byLevel);
    return prependToLines(block, indentation);
}

boost::format& cvx::str::details::formatImpl(boost::format& format)
{
    return format;
}

string cvx::str::toVariableName(string s)
{
    boost::regex disallowedChars("[^a-z0-9_]+");
    string res = boost::algorithm::to_lower_copy(s);
    res = boost::regex_replace(res, disallowedChars, string("_"));

    if (res[0]=='_')
    {
        res = 'v'+res;
    }

    return res;
}
