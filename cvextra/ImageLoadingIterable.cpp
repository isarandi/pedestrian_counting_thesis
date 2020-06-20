#include "ImageLoadingIterable.hpp"
#include "filesystem.hpp"
#include "vectors.hpp"
#include "io.hpp"

using namespace std;
using namespace cv;
using namespace cvx;

ImageLoadingIterable::
ImageLoadingIterable(
        vector<bpath> const& paths,
        int flags)
    : paths(paths)
    , flags(flags) {}

ImageLoadingIterable::
ImageLoadingIterable(
        bpath const& folderPath,
        boost::regex const& pattern,
        int flags)
    : flags(flags)
{
    paths = cvx::filesystem::listdir(folderPath, pattern);
}

auto ImageLoadingIterable::
operator[](int i) const -> Mat
{
    return cvx::imread(paths[i], flags);
}

auto ImageLoadingIterable::
begin() const -> iterator_t
{
    return boost::make_transform_iterator(paths.begin(),
        function<Mat(bpath const&)>(
            [this](bpath const& p)
            {
                return cvx::imread(p,flags);
            }));
}

auto ImageLoadingIterable::
end() const -> iterator_t
{
    return boost::make_transform_iterator(paths.end(),
        function<Mat(bpath const&)>(
            [this](bpath const& p)
            {
                return cvx::imread(p,flags);
            }));
}

size_t ImageLoadingIterable::
size() const
{
    return paths.size();
}


auto cvx::
imagesIn(bpath const& folderPath, std::string const& pattern, int flags) -> ImageLoadingIterable
{
    return ImageLoadingIterable{folderPath, boost::regex(pattern), flags};
}

auto cvx::
imagesIn(std::vector<bpath> const& folderPaths, std::string const& pattern, int flags) -> ImageLoadingIterable
{
    vector<bpath> filePaths;
    for (auto const& folderPath : folderPaths)
    {
        cvx::vectors::push_back_all(filePaths, cvx::filesystem::listdir(folderPath, pattern));
    }

    return ImageLoadingIterable{filePaths, flags};
}

auto ImageLoadingIterable::
range(
		int from,
		int to
		) const -> ImageLoadingIterable
{
	if (to == cvx::END)
	{
		to = paths.size();
	}

	return ImageLoadingIterable(cvx::vectors::subVector(paths, from, to));
}

auto ImageLoadingIterable::
range(
		cv::Range const& range
		) const -> ImageLoadingIterable {

    if (range == cv::Range::all())
    {
        return *this;
    }
	return this->range(range.start, range.end);
}

auto ImageLoadingIterable::
load() const -> std::vector<cv::Mat>
{
    vector<Mat> images;

    for (auto const& path : paths)
    {
        images.push_back(cvx::imread(path, flags));
    }

    return images;
}

auto cvx::loadImages(bpath const& folderPath, std::string const& pattern, int flags) -> vector<Mat>
{
    vector<Mat> images;

    for (auto const& path : cvx::filesystem::listdir(folderPath, pattern))
    {
        images.push_back(cvx::imread(path, flags));
    }

    return images;
}
