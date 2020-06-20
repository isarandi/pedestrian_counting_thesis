#ifndef FILTERBANK_HPP
#define FILTERBANK_HPP

#include <cvextra/core.hpp>
#include <CrowdCounting/RegionCounting/Features/FeatureExtractor.hpp>
#include <CrowdCounting/RegionCounting/Features/PreprocessedFrame.hpp>
#include <opencv2/core/core.hpp>
#include <stdx/cloning.hpp>
#include <memory>
#include <string>
#include <vector>

namespace crowd
{

class FilterBank : public FeatureExtractor
{
public:
    FilterBank(
            std::vector<cv::Mat> filters,
            std::vector<std::string> filterNames = std::vector<std::string>(),
            bool masked = false);

    virtual auto extract(PreprocessedFrame const& frame, cvx::Rectd relativeRect) const -> std::vector<double>;
    virtual auto getFeatureCount() const -> int;
    virtual auto getNames() const -> std::vector<std::string>;
    virtual auto getDescription() const -> std::string;

    auto getFilters() -> std::vector<cv::Mat> {return filters;}

    auto getResponses(cv::InputArray input) const -> std::vector<cv::Mat>;
    void getResponses(cv::InputArray input, std::vector<cv::Mat>& out) const;
    auto getAverageResponses(cv::InputArray input, cv::InputArray mask=cv::noArray()) const -> std::vector<double>;

    static auto LM(int oddSize) -> FilterBank;
    static auto CircLM(int oddSize) -> FilterBank;

    CVX_CLONE_IN_DERIVED(FilterBank)
    CVX_CONFIG_DERIVED(FilterBank)

private:
    std::vector<cv::Mat> filters;
    std::vector<std::string> filterNames;
    bool masked;

    static auto createLMFilters(int oddSize) -> std::vector<cv::Mat>;
    static auto createCircLMFilters(int oddSize) -> std::vector<cv::Mat>;
};

} //namespace crowd

#endif // FILTERBANK_HPP
