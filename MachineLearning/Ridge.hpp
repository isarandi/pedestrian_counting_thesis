#ifndef RIDGE_HPP
#define RIDGE_HPP

#include <MachineLearning/LearningSet.hpp>
#include <MachineLearning/Regression.hpp>
#include <opencv2/core/core.hpp>
#include <stdx/cloning.hpp>
#include <boost/property_tree/ptree.hpp>
//#include <Eigen/Core>
//#include <Eigen/Cholesky>
#include <armadillo>
#include <memory>
#include <string>
#include <vector>

namespace crowd {

class Ridge : public RegressionWithConfidence
{
public:

    /**
     * @param C Sets the tradeoff between fitting the data and regularization.
     * Large C <=> fit to data preferred, small C <=> strong regularization
     */
    Ridge(double C);
    Ridge(double C, double weightPriorVariance);

    // Regression interface
    virtual void train(LearningSet const& ls);
    virtual auto predict(cv::Mat1d const& X) const -> cv::Mat1d;
    virtual auto predictWithConfidence(cv::Mat1d const& X) const -> PredictionWithConfidence;

    virtual auto getJacobian(cv::Mat1d const& X) const -> cv::Mat1d;

    virtual auto getDescription() const -> std::string;

    auto getWeights() const -> cv::Mat1d{return *weights;}

    virtual auto describe() const -> boost::property_tree::ptree;
    static auto create(boost::property_tree::ptree const& pt) -> std::unique_ptr<Ridge>;

    CVX_CLONE_IN_DERIVED(Ridge)

private:
    stdx::cloned_unique_ptr<cv::Mat1d> weights;

    //typedef decltype(Eigen::MatrixXd().llt()) EigenLLT;
    //stdx::copy_constructed_unique_ptr<EigenLLT> llt; // = Cholesky((K+.)^-1)
    arma::mat L;

    double ridgeC;
    double weightPriorVariance;
};

}


#endif // RIDGE_HPP
