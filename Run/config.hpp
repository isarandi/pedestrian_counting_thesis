#ifndef CONFIG_HPP
#define CONFIG_HPP

#include <CrowdCounting/PersonLocations.hpp>
#include <cvextra/ImageLoadingIterable.hpp>
#include <cvextra/filesystem.hpp>
#include <cvextra/io.hpp>
#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/json_parser.hpp>

namespace config {

cvx::bpath const DATA_PATH{"/work/sarandi/crowd/data/"};
cvx::bpath const RESULT_PATH{"/work/sarandi/crowd/results/"};
cvx::bpath const SETUP_PATH{"/work/sarandi/crowd/setups/"};

inline
auto framePath(std::string const& name) -> cvx::bpath
{
    return DATA_PATH/"frames"/name;
}

inline
auto cannyPath(std::string const& name) -> cvx::bpath
{
    return DATA_PATH/"intermediate/canny"/name;
}

inline
auto dilatedCannyPath(std::string const& name) -> cvx::bpath
{
    return DATA_PATH/"intermediate/canny_dilated"/name;
}

inline
auto maskPath(std::string const& name) -> cvx::bpath
{
    return DATA_PATH/"intermediate/mask"/name;
}

inline
auto locationPath(std::string const& name) -> cvx::bpath
{
    return DATA_PATH/"ground_truth"/"person_instances"/(name + ".csv");
}

inline
auto roiStencilPath(std::string const& name) -> cvx::bpath
{
    return DATA_PATH/"other_input"/"roi_stencil"/name/"roi.png";
}

inline
auto configPath(std::string const& name, std::string const& aspect) -> cvx::bpath
{
    return DATA_PATH/"other_input"/"config"/(name+"_"+aspect+".json");
}

inline
auto scaleMapPath(std::string const& name) -> cvx::bpath
{
	return DATA_PATH/"other_input"/"scale_map"/(name+".csv");
}

inline
auto setupPath(std::string const& name) -> cvx::bpath
{
	return SETUP_PATH/(name+".json");
}

inline
auto textonPath(std::string const& name) -> cvx::bpath
{
	return DATA_PATH/"intermediate"/"texton"/name;
}

/////////////////////////////////////////////////////////////////////////////////

inline
auto frames(std::string const& name) -> cvx::ImageLoadingIterable
{
    return cvx::imagesIn(config::framePath(name));
}


inline
auto cannys(std::string const& name) -> cvx::ImageLoadingIterable
{
    return cvx::imagesIn(config::cannyPath(name), ".*png", cv::IMREAD_GRAYSCALE);
}

inline
auto dilatedCannys(std::string const& name) -> cvx::ImageLoadingIterable
{
    return cvx::imagesIn(config::dilatedCannyPath(name), ".*png", cv::IMREAD_GRAYSCALE);
}

inline
auto masks(std::string const& name) -> cvx::ImageLoadingIterable
{
    return cvx::imagesIn(config::maskPath(name), ".*png", cv::IMREAD_GRAYSCALE);
}

inline
auto locations(std::string const& name) -> crowd::PersonLocations
{
    return crowd::PersonLocations{config::locationPath(name)};
}

inline
auto roiStencil(std::string const& name) -> cvx::BinaryMat
{
    return cvx::imread(roiStencilPath(name), cv::IMREAD_GRAYSCALE);
}

inline
auto scaleMap(std::string const& name) -> cv::Mat1d
{
    return cvx::io::readDoubleMatFromCSV(scaleMapPath(name));
}

inline
auto textons(std::string const& name) -> cvx::ImageLoadingIterable
{
    return cvx::imagesIn(textonPath(name));
}


inline
auto frames_immediate(std::string const& name) -> std::vector<cv::Mat>
{
    return cvx::loadImages(config::framePath(name));
}

inline
auto cannys_immediate(std::string const& name) -> std::vector<cv::Mat>
{
    return cvx::loadImages(config::cannyPath(name), ".*png", cv::IMREAD_GRAYSCALE);
}

inline
auto masks_immediate(std::string const& name) -> std::vector<cv::Mat>
{
    return cvx::loadImages(config::maskPath(name), ".*png", cv::IMREAD_GRAYSCALE);
}

}

#endif // CONFIG_HPP
