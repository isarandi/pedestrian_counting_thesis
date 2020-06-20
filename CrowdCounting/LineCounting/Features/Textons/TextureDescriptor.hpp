#ifndef CROWDCOUNTING_LINECOUNTING_FEATURES_TEXTONS_TEXTUREDESCRIPTOR_HPP_
#define CROWDCOUNTING_LINECOUNTING_FEATURES_TEXTONS_TEXTUREDESCRIPTOR_HPP_

#include <CrowdCounting/RegionCounting/Features/Extractors/FilterBank.hpp>

#include <cvextra/core.hpp>
#include <stdx/cloning.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/objdetect/objdetect.hpp>

#include <memory>
#include <vector>

namespace crowd
{


class TextureDescriptor
{
public:
    virtual auto
    describePoints(
            cv::InputArray img,
            cv::InputArray stencil,
            cv::OutputArray outFeatures) -> std::vector<cv::Point> = 0;

    virtual auto
    getDescriptorSize() const -> int = 0;

    virtual ~TextureDescriptor(){}

    CVX_CLONE_IN_BASE(TextureDescriptor)
    CVX_CONFIG_BASE(TextureDescriptor)
};

class LocalHOGDescriptor : public TextureDescriptor
{
public:
    LocalHOGDescriptor(cv::Size cellSize, int nGradientBins)
        : cellSize(cellSize)
        , hogCalculator(cellSize*2, cellSize*2, cellSize, cellSize, nGradientBins)
        , nGradientBins(nGradientBins){}

    virtual auto
    describePoints(
            cv::InputArray img,
            cv::InputArray stencil,
            cv::OutputArray outFeatures) -> std::vector<cv::Point>;

    virtual auto
    getDescriptorSize() const -> int;

    CVX_CLONE_IN_DERIVED(LocalHOGDescriptor)
    CVX_CONFIG_DERIVED(LocalHOGDescriptor)

private:
    cv::Size cellSize;
    int nGradientBins;
    cv::HOGDescriptor hogCalculator;
};

class FilterBankDescriptor : public TextureDescriptor
{
public:
    FilterBankDescriptor(int lmSize)
        : filterBank(FilterBank::LM(lmSize))
        , lmSize(lmSize){}

    virtual auto
    describePoints(
            cv::InputArray img,
            cv::InputArray stencil,
            cv::OutputArray outFeatures) -> std::vector<cv::Point>;

    virtual auto
    getDescriptorSize() const -> int;

    CVX_CLONE_IN_DERIVED(FilterBankDescriptor)
    CVX_CONFIG_DERIVED(FilterBankDescriptor)

private:
    std::vector<cv::Mat> filterResponses;
    int lmSize;
    FilterBank filterBank;
};

} /* namespace crowd */

#endif /* CROWDCOUNTING_LINECOUNTING_FEATURES_TEXTONS_TEXTUREDESCRIPTOR_HPP_ */
