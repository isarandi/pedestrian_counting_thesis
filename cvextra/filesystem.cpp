#include "filesystem.hpp"

#include <boost/filesystem.hpp>
#include <boost/regex.hpp>

#include <fstream>
#include <vector>
#include <algorithm>

using namespace std;
using namespace cvx;

auto cvx::filesystem::
getFilePathsInFolder(
        bpath const& dirPath,
        boost::regex const& pattern) -> vector<string>
{
    vector<bpath> paths = listdir(dirPath, pattern);

    vector<string> strPaths;
    for (bpath const& p : paths)
    {
        strPaths.push_back(p.string());
    }

    return strPaths;
}

auto cvx::filesystem::
listSubfolders(bpath const& dirPath, boost::regex const& pattern) -> vector<bpath>
{
    vector<bpath> result;

    boost::filesystem::directory_iterator endIter;
    for (boost::filesystem::directory_iterator dirIter(dirPath);
         dirIter != endIter;
         ++dirIter)
    {
        if (boost::filesystem::is_directory(dirIter->status()))
        {
            string filename = dirIter->path().filename().string();

            if (boost::regex_match(filename, pattern))
            {
                result.push_back(dirIter->path());
            }
        }
    }

    std::sort(result.begin(), result.end());
    return result;
}

auto cvx::filesystem::
listdir(
        bpath const& dirPath,
        boost::regex const& pattern
        ) -> vector<bpath>
{
    vector<bpath> result;
    if (!boost::filesystem::exists(dirPath))
	{
    	return result;
	}

    boost::filesystem::directory_iterator endIter;
    for (boost::filesystem::directory_iterator dirIter(dirPath);
         dirIter != endIter;
         ++dirIter)
    {
        if (boost::filesystem::is_regular_file(dirIter->status()))
        {
            string filename = dirIter->path().filename().string();

            if (boost::regex_match(filename, pattern))
            {
                result.push_back(dirIter->path());
            }
        }
    }

    std::sort(result.begin(), result.end());

    return result;
}

auto cvx::filesystem::
listdir(
        bpath const& dirPath,
        std::string const& pattern
        ) -> std::vector<bpath>
{
    return listdir(dirPath, boost::regex(pattern));
}


auto cvx::filesystem::
listSubfolders(
        bpath const& dirPath,
        std::string const& pattern
        )-> std::vector<bpath>
{
    return listSubfolders(dirPath, boost::regex(pattern));
}
