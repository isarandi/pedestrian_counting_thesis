#ifndef PERSISTENCE_HPP
#define PERSISTENCE_HPP

#include <boost/filesystem.hpp>
#include <cvextra/filesystem.hpp>
#include <cvextra/io.hpp>
#include <cvextra/strings.hpp>
#include <opencv2/core/core.hpp>
#include <functional>
#include <map>
#include <string>
#include <vector>

class Persistence
{
public:
    static
    void off();

    static
    void saveMat(std::string const& name, cv::Mat const& mat);

    static
    void save(std::string const& name, cv::Mat const& mat);

    static
    void save(std::string const& name, std::vector<cv::Mat> const& mats);

    static
    auto canLoad(std::string const& name) -> bool;

    template <typename T>
    static
    auto loadOrCompute(std::string const& name, std::function<T(void)> createDataFunc) -> T
    {
        if (canLoad(name))
        {
            return loadData<T>(name);
        } else {
            T data = createDataFunc();
            saveData(name, data);
            return data;
        }
    }

    static
    auto loadOrComputeMat(
            std::string const& name,
            std::function<cv::Mat(void)> createDataFunc,
            bool forceComputation = false,
            bool forceMultiple = false
            ) -> cv::Mat;

    static
    auto loadOrComputeMats(
            std::string const& name,
            std::vector<std::string> const& suffixes,
            std::function<std::vector<cv::Mat>(void)> createMatsFunc,
            bool forceComputation = false,
            bool forceMultiple = false
            ) -> std::vector<cv::Mat>;

    static
    auto loadMat(std::string const& name) -> cv::Mat;
    static
    auto loadMats(std::string const& name) -> std::vector<cv::Mat>;

    template <typename T>
    static
    auto loadData(std::string const& name) -> T
	{
    	return getInstance()._loadData<T>(name);
    }

    template <typename T>
    static
    void saveData(std::string const& name, T const& data)
	{
    	return getInstance()._saveData(name, data);
    }

private:
    Persistence();
    virtual ~Persistence();

    static auto getInstance() -> Persistence&;


    void _save(std::string const& name, std::vector<cv::Mat> const& mats);
    auto _loadMats(std::string const& name) -> std::vector<cv::Mat>;

    auto _canLoad(std::string const& name) -> bool;

    void _saveMat(std::string const& name, cv::Mat const& mat);
    auto _loadMat(std::string const& name) -> cv::Mat;


    template <typename T>
    auto _loadData(std::string const& name) -> T
    {
    	cvx::bpath filePath = nameToFilePath[name];
    	return cvx::io::readFileStorage<T>(filePath);
    }

    template <typename T>
    void _saveData(std::string const& name, T const& data)
    {
        #pragma omp critical
        {
            cvx::bpath filePath = dataFolder/cvx::str::format("%10d.yml.gz", counter);
            ++counter;
            nameToFilePath[name] = filePath;
            cvx::io::writeFileStorage(filePath, data);
            saveCatalog();
        }
    }

    void saveCatalog();
    void loadCatalog();

    bool enabled;
    cvx::bpath catalogPath;
    cvx::bpath dataFolder;
    int counter;
    std::map<std::string, cvx::bpath> nameToFilePath;
};


#endif // PERSISTENCE_HPP
