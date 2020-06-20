#ifndef FILEUTILS_HPP
#define FILEUTILS_HPP

#include <boost/filesystem.hpp>
#include <boost/regex.hpp>
#include <string>
#include <vector>

namespace cvx {

typedef boost::filesystem::path bpath;

namespace filesystem
{

auto getFilePathsInFolder(
        bpath const& dirPath,
        boost::regex const& pattern=boost::regex(".*")
        ) -> std::vector<std::string>;

auto listdir(
        bpath const& dirPath,
        boost::regex const& pattern=boost::regex(".*")
        ) -> std::vector<bpath>;

auto listSubfolders(
        bpath const& dirPath,
        boost::regex const& pattern=boost::regex(".*")
        ) -> std::vector<bpath>;

auto listdir(
        bpath const& dirPath,
        std::string const& pattern
        ) -> std::vector<bpath>;


auto listSubfolders(
        bpath const& dirPath,
        std::string const& pattern
        )-> std::vector<bpath>;

}

}

#endif // FILEUTILS_HPP
