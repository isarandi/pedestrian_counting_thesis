#ifndef COUNTINGTESTCASE_HPP
#define COUNTINGTESTCASE_HPP

#include <cvextra/utils.hpp>
#include <CrowdCounting/RegionCounting/FrameCollection.hpp>
#include <CrowdCounting/RegionCounting/RegionCounter.hpp>
#include <stdx/cloning.hpp>
#include <memory>
#include <string>

namespace crowd
{
class CountingTestResult;
} /* namespace crowd */

namespace crowd {

class CountingTestCase
{
public:
    CountingTestCase(
            std::string const& name,
            FrameCollection const& training,
            FrameCollection const& test,
            RegionCounter const& counter)
        : name(name)
        , training(training)
        , test(test)
        , counter(counter.clone()){}

    CountingTestCase(
            FrameCollection const& training,
            FrameCollection const& test,
            RegionCounter const& counter)
        : name(cvx::timestamp())
        , training(training)
        , test(test)
        , counter(counter.clone()){}

    auto run() -> CountingTestResult;

    auto getTrainingSet() const -> FrameCollection const& {return training;}
    auto getTestSet() const -> FrameCollection const& {return test;}
    auto getCrowdCounter() const -> RegionCounter const& {return *counter;}
    auto getName() const -> std::string {return name;}

    auto getDescription() const -> std::string;

    CVX_CLONE_IN_SINGLE(CountingTestCase)

private:
    std::string name;
    FrameCollection training;
    FrameCollection test;
    std::shared_ptr<RegionCounter> counter;
};

}

#endif // COUNTINGTESTCASE_HPP
