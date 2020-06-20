#ifndef CROWDCOUNTING_LINECOUNTING_FEATURES_TEXTONS_TEXTONCREATOR_HPP_
#define CROWDCOUNTING_LINECOUNTING_FEATURES_TEXTONS_TEXTONCREATOR_HPP_

#include <cvextra/ImageLoadingIterable.hpp>
#include <stdx/cloning.hpp>
#include <CrowdCounting/RegionCounting/Features/Extractors/FilterBank.hpp>
#include <MachineLearning/DataNormalizer.hpp>
#include <CrowdCounting/LineCounting/Features/Textons/TextureDescriptor.hpp>
#include <opencv2/core/core.hpp>
#include <vector>

namespace crowd {
namespace linecounting {

namespace details {
    struct SampleInfo{
        int iFrame;
        cv::Point p;
        bool mirrored;
    };
}

class TextonCreator {
public:
    enum class Descriptor{LM_FILTERS, LOCAL_HOG, BOTH};

	TextonCreator(
	        cv::Size processingSize,
	        int nTextons,
	        TextureDescriptor const& descriptor,
	        DataNormalizer const& normalizer,
	        bool mirroredAlso);

    TextonCreator(
            cv::Size processingSize,
            int nTextons,
            TextureDescriptor const& descriptor,
            bool mirroredAlso);

	void train(cvx::ImageLoadingIterable const& images, cvx::BinaryMat const& stencil, std::string const& name);
	void trainCompute(cvx::ImageLoadingIterable const& images, cvx::BinaryMat const& stencil);

	auto getTextonMap(cv::Mat3b const& image) -> cv::Mat1b;

	cv::Size processingSize;
	cv::Mat1f textonCenters;

	stdx::cloned_unique_ptr<TextureDescriptor> textureDescriptor;

	int nTextons;
	bool mirroredAlso;

	CVX_CONFIG_SINGLE(TextonCreator)

private:
	void savePrototypicalTextonInstances(
	        cv::Mat1f const& foregroundDescriptors,
	        std::vector<details::SampleInfo> const& sampleInfos,
	        cv::Mat1f const& textonCenters,
	        cvx::ImageLoadingIterable const& images) const;

	void illustrateTextonMaps(
            std::vector<details::SampleInfo> const& sampleInfos,
            cv::Mat1b const& textonLabels,
            cvx::ImageLoadingIterable const& images) const;

	stdx::cloned_unique_ptr<DataNormalizer> normalizer;


};

} /* namespace linecounting */
} /* namespace crowd */

#endif /* CROWDCOUNTING_LINECOUNTING_FEATURES_TEXTONS_TEXTONCREATOR_HPP_ */
