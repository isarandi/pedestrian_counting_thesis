#ifndef IOUTILS_HPP
#define IOUTILS_HPP

#include "filesystem.hpp"
#include <string>
#include <opencv2/opencv.hpp>
#include <boost/filesystem.hpp>
#include <chrono>
#include <fstream>

namespace cvx {

auto imread(bpath const& p, int flags = 1) -> cv::Mat;
void imwrite(
        bpath const& p,
        cv::InputArray src,
        std::vector<int> params = std::vector<int>());

void writeVideo(
        bpath const& targetFilePath,
        std::vector<cv::Mat> const& frames,
        double framesPerSecond);

void writeVideo(
        bpath const& targetFilePath,
        std::vector<bpath> const& framePaths,
        double framesPerSecond);

#define CVX_FOURCC_MACRO(c1,c2,c3,c4) ((c1 & 255) + ((c2 & 255) << 8) + ((c3 &255) << 16) + ((c4 & 255) << 24))
enum FourCC
{
    XVID = CVX_FOURCC_MACRO('X', 'V', 'I', 'D'),
    FFV1 = CVX_FOURCC_MACRO('F', 'F', 'V', 'I'),
};

namespace io {

void statusUpdate(
        std::string const& msg,
        std::chrono::duration<long> timeInterval = std::chrono::seconds(2));

void saveToMatlab(
        cv::Mat mat,
        std::string const& name,
        bpath const& folder);

void saveAsMatlabMat(
        cv::Mat const& mat,
        std::string const& name,
        bpath const& folder);

template <typename T>
void write(bpath const& path, T const& content)
{
    boost::filesystem::create_directories(path.parent_path());
    std::ofstream file(path.string());
    file << content;
}

inline
auto readFile(bpath const& path) -> std::string
{
    std::stringstream ss;
    std::ifstream file(path.string());
    ss << file.rdbuf();
    return ss.str();
}

void writeMatToBinaryFile(bpath const& path, cv::Mat mat);
auto readMatFromBinaryFile(bpath const& path) -> cv::Mat;

auto readDoubleMatFromCSV(bpath const& path) -> cv::Mat1d;
void writeToCSV(bpath const& path, cv::Mat1d const& m);

template<typename T>
auto binaryWrite(std::ostream& stream, T value) -> std::ostream&
{
    return stream.write(reinterpret_cast<char const*>(&value), sizeof(T));
}

template<typename T>
auto binaryRead(std::istream& stream) -> T
{
    T value;
    stream.read(reinterpret_cast<char*>(&value), sizeof(T));
    return value;
}

auto linesOf(bpath const& path) -> std::vector<std::string>;

template <typename T>
void writeFileStorage(bpath const& path, T const& content, std::string const& nodeName="data")
{
    boost::filesystem::create_directories(path.parent_path());
    cv::FileStorage fileStorage(path.string(), cv::FileStorage::WRITE);
    fileStorage << nodeName << content;
}

template <typename T>
auto readFileStorage(bpath const& path, std::string const& nodeName="data") -> T
{
    cv::FileStorage fileStorage(path.string(), cv::FileStorage::READ);
    T result;
    fileStorage[nodeName] >> result;
    return result;
}


template <typename T>
auto fsget(cv::FileStorage const& fileStorage, std::string const& nodeName) -> T
{
    T result;
    fileStorage[nodeName] >> result;
    return result;
}




}}

#endif // IOUTILS_HPP
