#include "io.hpp"
#include "strings.hpp"
#include "utils.hpp"
#include "mats.hpp"
#include "vectors.hpp"
#include "filesystem.hpp"
#include <boost/algorithm/string.hpp>
#include <sstream>
#include <iostream>
#include <fstream>
#include <chrono>

using namespace std;
using namespace cv;
using namespace boost::filesystem;
using namespace std::chrono;

void cvx::io::
statusUpdate(
        string const& msg,
        duration<long> timeInterval)
{
    static system_clock::time_point lastTime;

    if (system_clock::now()-lastTime > timeInterval)
    {
        // Make sure that the line end is added into the stream
        // at the same time as the message.

        stringstream ss;
        ss << "\r" << msg;
        cout << ss.str();
        cout.flush();

        lastTime = system_clock::now();
    }
}

void cvx::io::
saveToMatlab(Mat mat, string const& name, bpath const& folder)
{
    if (!boost::filesystem::exists(folder))
    {
        boost::filesystem::create_directories(folder);
    }

    string filename = cvx::timestamp() + "_" + name + ".m";
    ofstream outFile((folder / filename).string());
    outFile << name << " = " << mat << ";";
}

auto cvx::io::
linesOf(bpath const& path) -> vector<string>
{
    ifstream in{path.string()};
    vector<string> result;

    string line;
    while (std::getline(in, line))
    {
        result.push_back(line);
    }

    return result;
}

void cvx::
imwrite(bpath const& p, InputArray src, vector<int> params)
{
    if (!boost::filesystem::exists(p.parent_path()))
    {
        boost::filesystem::create_directories(p.parent_path());
    }
    cv::imwrite(p.string(), src, params);
}

auto cvx::
imread(bpath const& p, int flags) -> Mat
{
    return cv::imread(p.string(), flags);
}

void cvx::
writeVideo(bpath const& filePath, vector<Mat> const& frames, double framesPerSecond)
{
    CV_Assert(!frames.empty());

    VideoWriter videoWriter{filePath.string(), FourCC::XVID, framesPerSecond, frames[0].size()};

    for (auto& frame : frames)
    {
        videoWriter << frame;
    }
}

void cvx::
writeVideo(
		bpath const& targetFilePath,
		vector<bpath> const& framePaths,
		double framesPerSecond)
{
    CV_Assert(!framePaths.empty());

    VideoWriter videoWriter{targetFilePath.string(), FourCC::XVID, framesPerSecond, cvx::imread(framePaths[0]).size()};

    for (auto& framePath : framePaths)
    {
        videoWriter << cvx::imread(framePath);
    }
}

void cvx::io::
writeMatToBinaryFile(bpath const& path, cv::Mat mat)
{
    std::ofstream stream{path.string(), std::ios::binary};

    binaryWrite(stream, mat.size());
    binaryWrite(stream, mat.type());

    cv::Mat continuousMat = (mat.isContinuous() ? mat : mat.clone());
    stream.write(
            reinterpret_cast<const char*>(continuousMat.datastart),
            continuousMat.dataend-continuousMat.datastart);
}

auto cvx::io::
readMatFromBinaryFile(const bpath& path) -> Mat
{
    std::ifstream stream{path.string(), std::ios::binary};

    Size size = binaryRead<Size>(stream);
    int type = binaryRead<int>(stream);

    Mat mat{size, type};
    stream.read(reinterpret_cast<char*>(mat.datastart), mat.dataend-mat.datastart);
    return mat;
}

auto cvx::io::
readDoubleMatFromCSV(bpath const& path) -> Mat1d
{
	Mat1d result;
	auto lines = cvx::io::linesOf(path);

	for (auto const& line : lines)
	{
		if (boost::algorithm::starts_with(line, "#") || line.empty())
		{
			continue;
		}
		auto parts =
				cvx::vectors::transform(
						cvx::str::split(line, ","),
						[](std::string const& s){return std::stod(s);});

		Mat1d asMat = cvx::mats::matFromRows({parts});
		if (result.empty())
		{
			result = asMat;
		} else {
			cvx::vconcat(result, asMat, result);
		}
	}
	return result;
}

void cvx::io::
writeToCSV(bpath const& path, cv::Mat1d const& m)
{
	boost::filesystem::create_directories(path.parent_path());
	ofstream out{path.string()};

	for (int iRow = 0; iRow < m.rows; ++iRow)
	{
		for (int iCol = 0; iCol < m.cols; ++iCol)
		{
			out << m(iRow, iCol);
			if (iCol < m.cols-1)
			{
				out << ",";
			}
		}
		if (iRow < m.rows-1)
		{
			out << "\n";
		}
	}
}
