#ifndef MACHINELEARNING_DATANORMALIZER_HPP_
#define MACHINELEARNING_DATANORMALIZER_HPP_

#include <cvextra/configfile.hpp>
#include <opencv2/core/core.hpp>
#include <stdx/cloning.hpp>
#include <memory>
#include <string>

namespace crowd
{

class DataNormalizer
{
public:
    virtual void
    train(cv::InputArray data) = 0;

    virtual void
    apply(cv::InputArray src, cv::OutputArray) const = 0;

    virtual auto
    getJacobian(cv::InputArray singleInput) const -> cv::Mat = 0;

    auto
    applyRet(cv::InputArray src) -> cv::Mat const;
    virtual ~DataNormalizer(){}

    virtual void
    savePersistence(std::string const& name) const = 0;

    virtual void
    loadPersistence(std::string const& name) = 0;

    CVX_CLONE_IN_BASE(DataNormalizer)
    CVX_CONFIG_BASE(DataNormalizer)
};

class PerFeatureNormalizer : public DataNormalizer
{
public:
    virtual void
    train(cv::InputArray data);

    virtual void
    apply(cv::InputArray src, cv::OutputArray) const;

    virtual auto
    getJacobian(cv::InputArray singleInput) const -> cv::Mat;

    virtual void
    savePersistence(std::string const& name) const;

    virtual void
    loadPersistence(std::string const& name);

    CVX_CLONE_IN_DERIVED(PerFeatureNormalizer)
    CVX_CONFIG_DERIVED(PerFeatureNormalizer)

private:
    cv::Mat means; //row mat
    cv::Mat stdevs; //row mat

};

class Whitener : public DataNormalizer
{
public:
    virtual void
    train(cv::InputArray data);

    virtual void
    apply(cv::InputArray src, cv::OutputArray) const;

    virtual auto
    getJacobian(cv::InputArray singleInput) const -> cv::Mat;

    virtual void
    savePersistence(std::string const& name) const;

    virtual void
    loadPersistence(std::string const& name);

    CVX_CLONE_IN_DERIVED(Whitener)
    CVX_CONFIG_DERIVED(Whitener)

private:
    cv::Mat means; //row mat
    cv::Mat weights;

};



} /* namespace crowd */

#endif /* MACHINELEARNING_DATANORMALIZER_HPP_ */
