#ifndef ILLUSTRATE_PLOTS_HPP_
#define ILLUSTRATE_PLOTS_HPP_

#include <cvextra/cvret.hpp>
#include <cvextra/vectors.hpp>
#include <opencv2/core/core.hpp>
#include <Python/EasyCallable.hpp>
#include <Python/EasyObject.hpp>
#include <Python/Pyplot.hpp>
#include <string>
#include <vector>

namespace crowd {

inline
void plotWithStdev(pyx::Pyplot const& plt, cv::InputArray mean, cv::InputArray stdev, std::string const& meanFormat)
{
    plt.plot(mean, meanFormat);
    plt.fill_between(
                {   cvx::vectors::range(mean.size().height),
                    cvret::subtract(mean,stdev),
                    cvret::add(mean,stdev)},"alpha=0.1");
}


template <typename T>
void plotWithStdevAx(T ax, cv::InputArray mean, cv::InputArray stdev, std::string const& meanFormat)
{
    ax.call("plot", {mean, meanFormat});
    ax.call("fill_between",
                {   cvx::vectors::range(mean.size().height),
                    cvret::subtract(mean,stdev),
                    cvret::add(mean,stdev)},"alpha=0.1");
}

} /* namespace crowd */

#endif /* ILLUSTRATE_PLOTS_HPP_ */
