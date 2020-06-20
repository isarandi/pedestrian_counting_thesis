#ifndef BACKGROUNDSEGMENTER_HPP
#define BACKGROUNDSEGMENTER_HPP

#include <cvextra/core.hpp>
#include <cvextra/improc.hpp>
#include <opencv2/core/core.hpp>
#include <stdx/cloning.hpp>
#include <memory>

namespace crowd { namespace bg {

/**
 * @brief Performs foreground/background segmentation of a frame sequence.
 * Abstract class.
 */
class BackgroundModel
{
public:

    virtual void segmentNext(cv::InputArray nextImage, cv::OutputArray foregroundMask) = 0;
    virtual auto segmentNextRet(cv::InputArray nextImage) -> cvx::BinaryMat;


    virtual void initialize(cv::Size processingSize) = 0;

    template <typename Images>
    void initializeWithImages(cv::Size procSize, Images images)
    {
        initialize(procSize);

        cv::Mat result;
        cv::Mat resizedImage;

        for (auto const& im : images)
        {
            cvx::resizeBest(im, resizedImage, procSize);
            segmentNext(resizedImage, result);
        }
    }

    virtual ~BackgroundModel(){}

    CVX_CLONE_IN_BASE(BackgroundModel)
};

}}

#endif // BACKGROUNDSEGMENTER_HPP
