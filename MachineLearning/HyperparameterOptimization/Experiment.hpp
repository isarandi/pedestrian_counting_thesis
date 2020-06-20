#ifndef EXPERIMENT_HPP
#define EXPERIMENT_HPP

#include <opencv2/opencv.hpp>
#include <vector>
#include <string>

namespace crowd {

class Experiment
{
public:

    /**
     * Returns an error measure for the current parameters.
     */
    virtual auto evaluate(std::vector<double> const& parameters) const -> double = 0;
};

}

#endif // EXPERIMENT_HPP
