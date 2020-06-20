#ifndef DATASETS_HPP
#define DATASETS_HPP

#include <map>
#include <memory>
#include <string>

namespace crowd
{
class FrameCollection;
} /* namespace crowd */

namespace crowd
{
    auto getDatasets() -> std::map<std::string, std::shared_ptr<FrameCollection>>;
}

#endif // DATASETS_HPP
