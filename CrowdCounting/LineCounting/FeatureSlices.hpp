#ifndef CROWDCOUNTING_LineCounting_FEATURESLICES_HPP_
#define CROWDCOUNTING_LineCounting_FEATURESLICES_HPP_

#include <cvextra/core.hpp>
#include <cvextra/ImageLoadingIterable.hpp>
#include <opencv2/core/core.hpp>
#include <string>

namespace cvx {
class LineSegment;
} /* namespace cvx */


namespace crowd {

class FeatureSlices
{
public:
	cv::Mat3b image;
	cv::Mat2d flow;
	cv::Mat1b directionLabels;
	cvx::BinaryMat canny;
	cv::Mat2d grad;
	cvx::BinaryMat foregroundMask;
	cvx::BinaryMat stencil;
	cv::Mat1b textonMap;
	cv::Mat1b mirroredTextonMap;
	cv::Mat1d scaleMap;
	int nTextons;

	auto operator () (
	        cv::Rect const& rect
	        ) const -> FeatureSlices;
    auto operator () (
            cv::Range const& rowRange,
            cv::Range const& colRange
            ) const -> FeatureSlices;

    auto size() const -> cv::Size {return image.size();}
    auto empty() const -> bool {return image.empty();}
    auto horizontalFlipped() const -> FeatureSlices;

    void loadOrComputeTextons(
            std::string const& datasetName,
            std::string const& fullName,
            cvx::LineSegment const& seg);

    static
    auto loadOrCompute(
            std::string const& datasetName,
            cvx::LineSegment const& seg
            ) -> FeatureSlices;
};

struct CannyOptions {
	double gaussRadius;
	double threshold1;
	double threshold2;
	int dilateSize;
	int minEdgeLength;

    void writeToFile(cvx::bpath const& path);
    static auto fromFile(cvx::bpath const& path) -> CannyOptions;
};

auto createCannySlice(
		cvx::ImageLoadingIterable const& images,
		cvx::LineSegment const& seg,
		CannyOptions const& options
		) -> cvx::BinaryMat;

auto createGradientSlice(
		cvx::ImageLoadingIterable const& images,
		cvx::LineSegment const& seg,
		int dilateSize
		) -> cv::Mat2d;

}

#endif /* CROWDCOUNTING_LineCounting_FEATURESLICES_HPP_ */
