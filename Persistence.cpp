#include <boost/filesystem.hpp>
#include <opencv2/core/core.hpp>
#include <Persistence.hpp>
#include <utility>

using namespace std;
using namespace cv;
using namespace cvx;
using namespace boost::filesystem;

Persistence::
Persistence()
{
    enabled = true;
    catalogPath = "/work/sarandi/crowd/persistence/catalog.yml";
    dataFolder = "/work/sarandi/crowd/persistence/data";
    boost::filesystem::create_directories(dataFolder);

    loadCatalog();
}

Persistence::
~Persistence()
{
    saveCatalog();
}

auto Persistence::
getInstance() -> Persistence&
{
    static Persistence persistence;
    return persistence;
}

void Persistence::
_save(string const& name, vector<Mat> const& mats)
{
    #pragma omp critical
    {
        bpath filePath = dataFolder/cvx::str::format("%10d.yml", counter);
        ++counter;
        nameToFilePath[name] = filePath;

        FileStorage fs(filePath.string(), FileStorage::WRITE);
        fs << "mats" << "[";

        for (Mat const& mat : mats)
        {
            fs << mat;
        }
        fs << "]";

        saveCatalog();
    }
}

auto Persistence::
_canLoad(string const& name) -> bool
{
    return nameToFilePath.count(name)>0;
}


auto Persistence::
_loadMats(string const& name) -> vector<Mat>
{
	bpath filePath = nameToFilePath[name];
    FileStorage fs(filePath.string(), FileStorage::READ);
    vector<Mat> result;
    fs["mats"] >> result;
    return result;
}

auto Persistence::
_loadMat(std::string const& name) -> cv::Mat
{
    std::string hashedName = cvx::str::replace(name, "\\\"\\'[]{}", "");
    cvx::bpath filePath = nameToFilePath[hashedName];
    return cvx::io::readMatFromBinaryFile(filePath);
}

void Persistence::
_saveMat(std::string const& name, cv::Mat const& data)
{
    std::string hashedName = cvx::str::replace(name, "\\\"\\'[]{}", "");

    #pragma omp critical
    {
        if (nameToFilePath.count(hashedName)>0)
        {
            boost::filesystem::remove(nameToFilePath[hashedName]);
            nameToFilePath.erase(hashedName);
            saveCatalog();
        }

        cvx::bpath filePath = dataFolder/boost::filesystem::unique_path(cvx::str::format("%010d_%%%%%%%%.cvmat", counter));
        ++counter;
        nameToFilePath[hashedName] = filePath;

        cvx::io::writeMatToBinaryFile(filePath, data);
        saveCatalog();
    }
}


void Persistence::
saveCatalog()
{
    FileStorage fs(catalogPath.string(), FileStorage::WRITE);
    fs << "files" << "[";

    for (auto& elem : nameToFilePath)
    {
        fs << "{" << "name" << elem.first << "filePath" << elem.second.string() << "}";
    }

    fs << "]";
}

void Persistence::
loadCatalog()
{
    nameToFilePath.clear();

    FileStorage fs(catalogPath.string(), FileStorage::READ);
    FileNode catalogData = fs["files"];

    int largestStoredIndex = -1;
    for(auto const& elem : catalogData)
    {
        string name = (string) elem["name"];
        name = cvx::str::replace(name, "\\\"\\'[]{}", "");
        string filePath = (string) elem["filePath"];
        if (boost::filesystem::exists(filePath))
        {
            nameToFilePath[name] = filePath;

            int num = std::stoi(bpath(filePath).stem().string());
            largestStoredIndex = std::max(largestStoredIndex, num);
        }
    }
    counter = largestStoredIndex+1;
}

void Persistence::
off()
{
    getInstance().enabled = false;
}

void Persistence::
save(string const& name, Mat const& mat)
{
    getInstance()._saveMat(name, mat);
}

void Persistence::
saveMat(string const& name, Mat const& mat)
{
    getInstance()._saveMat(name, mat);
}

void Persistence::
save(string const& name, vector<Mat> const& mats)
{
    getInstance()._save(name, mats);
}

bool Persistence::
canLoad(string const& name)
{
    return getInstance().enabled && getInstance()._canLoad(name);
}

auto Persistence::
loadMat(string const& name) -> Mat
{
    return getInstance()._loadMat(name);
}

auto Persistence::
loadMats(string const& name) -> vector<Mat>
{
    return getInstance()._loadMats(name);
}

auto Persistence::
loadOrComputeMat(
        std::string const& name,
        std::function<cv::Mat(void)> createMatFunc,
        bool forceComputation,
        bool forceMultiple
        ) -> cv::Mat
{
    static set<std::string> has_been_recomputed;
    std::string hashedName = cvx::str::replace(name, "\\\"\\'[]{}", "");

    bool needsToCompute = !canLoad(hashedName) || (forceComputation && (forceMultiple || has_been_recomputed.count(hashedName)==0));

    if (needsToCompute && forceComputation)
    {
        cout << "Forced computation of " << name << endl;
        has_been_recomputed.insert(hashedName);
    }

    if (!needsToCompute)
    {
        return loadMat(hashedName);
    } else {
        cv::Mat mat = createMatFunc();
        saveMat(hashedName, mat);

        return mat;
    }
}

auto Persistence::
loadOrComputeMats(
        std::string const& name,
        std::vector<std::string> const& suffixes,
        std::function<std::vector<cv::Mat>(void)> createMatsFunc,
        bool forceComputation,
        bool forceMultiple
        ) -> std::vector<cv::Mat>
{
    static set<std::string> has_been_recomputed;
    std::string hashedName = cvx::str::replace(name, "\\\"{}[]()", "");
    vector<string> hashedSuffixes;

    for (string const& s : suffixes)
    {
        hashedSuffixes.push_back(cvx::str::replace(s, "\\\"{}[]()", ""));
    }

    bool needsToCompute = !canLoad(hashedName+hashedSuffixes[0]) || (forceComputation && (forceMultiple || !(has_been_recomputed.count(hashedName)>0)));

    if (needsToCompute && forceComputation)
    {
        cout << "Forced computation of " << name << endl;
        has_been_recomputed.insert(hashedName);
    }

    if (!needsToCompute)
    {
        std::vector<cv::Mat> results;
        for (auto const& hashedSuffix : hashedSuffixes)
        {
            results.push_back(loadMat(hashedName+hashedSuffix));
        }
        return results;

    } else {
        std::vector<cv::Mat> results = createMatsFunc();
        for (int i = 0; i<suffixes.size(); ++i)
        {
            saveMat(hashedName+hashedSuffixes[i], results[i]);
        }
        return results;
    }
}


