#ifndef PIXELMODEL_HPP
#define PIXELMODEL_HPP

#include <stdx/cloning.hpp>
#include <memory>
#include <random>
#include <vector>

namespace crowd { namespace bg {

/**
 * @brief Models the likelihood of an n dimensional feature vector
 * by kernel density estimation using an isotropic Gaussian kernel
 */
class LikelihoodModel
{
    typedef std::vector<double> Sample;

public:
    LikelihoodModel(double kernelSize, int maxWindowSize);

    /**
     * @param sample A new sample for which we want to estimate the likelihood
     * @return p(sample|model)
     */
    auto getLikelihood(Sample const& sample) const -> double;

    /**
     * @brief Updates the model by adding a new sample
     */
    void update(Sample const& sample);

    auto getSamples() const -> std::vector<Sample> const& {return samples;}

    CVX_CLONE_IN_SINGLE(LikelihoodModel)

private:

    std::vector<Sample> samples;

    /**
     * @brief Variance of the Gaussian kernel.
     * (Covariance assumed to be diagonal with equal elements)
     */
    double kernelSize;

    /**
     * @brief Maximum number of stored samples from which we estimate likelihoods
     */
    int maxWindowSize;

    std::default_random_engine randEng;

};

}}

#endif // PIXELMODEL_HPP
