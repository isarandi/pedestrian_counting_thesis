#include <CrowdCounting/LineCounting/Features/Textons/TextonCreator.hpp>
#include <Illustrate/illustrate.hpp>
#include <cvextra/core.hpp>
#include <cvextra/cvret.hpp>
#include <cvextra/utils.hpp>
#include <cvextra/improc.hpp>
#include <cvextra/LoopRange.hpp>
#include <Run/config.hpp>
#include <opencv2/core/mat.hpp>
#include <opencv2/core/types_c.h>
#include <opencv2/imgproc/imgproc.hpp>
#include <Python/Pyplot.hpp>
#include <Persistence.hpp>
#include <queue>
#include <utility>
#include <omp.h>


using namespace cv;
using namespace std;
using namespace cvx;
using namespace crowd;
using namespace crowd::linecounting;
using namespace crowd::linecounting::details;

TextonCreator::
TextonCreator(
        cv::Size processingSize,
        int nTextons,
        TextureDescriptor const& descriptor,
        DataNormalizer const& normalizer,
        bool mirroredAlso)
	: processingSize(processingSize)
	, nTextons(nTextons)
    , textureDescriptor(descriptor)
    , normalizer(normalizer)
    , mirroredAlso(mirroredAlso)
{
}

TextonCreator::
TextonCreator(
        cv::Size processingSize,
        int nTextons,
        TextureDescriptor const& descriptor,
        bool mirroredAlso)
    : processingSize(processingSize)
    , nTextons(nTextons)
    , textureDescriptor(descriptor)
    , mirroredAlso(mirroredAlso)
{
}

void TextonCreator::
train(ImageLoadingIterable const& images, BinaryMat const& stencil_, string const& dataName)
{
    string name = cvx::configfile::json_str(describe())+"_"+dataName;
    textonCenters = Persistence::loadOrComputeMat(name+"centers", [&](){
        trainCompute(images, stencil_);
        normalizer->savePersistence(name+"_normalizer");
        return textonCenters;
    });

    normalizer->loadPersistence(name+"_normalizer");
}

void TextonCreator::
trainCompute(ImageLoadingIterable const& images, BinaryMat const& stencil_)
{
	BinaryMat stencil = cvret::resize(stencil_, processingSize, 0, 0, INTER_NEAREST);

	int nFeatures = textureDescriptor->getDescriptorSize();
    Mat1f foregroundDescriptors{images.size()*processingSize.area()*2, nFeatures};
    vector<SampleInfo> sampleInfos;

    auto mirrorOptions = (mirroredAlso ? std::vector<bool>{false, true} : std::vector<bool>{false});

    //#pragma omp parallel for
	for (int iFrame : cvx::irange(images.size()))
	{
	    for (bool mirrored : mirrorOptions)
	    {
            cout << iFrame << endl;
            Mat3b image = (mirrored ? cvret::flip(images[iFrame], 1) : images[iFrame]);
            cv::resize(image, image, processingSize);
            //Mat3b labImage = cvret::cvtColor(image, COLOR_BGR2Lab);
            Mat1b grayImage = cvret::cvtColor(image, COLOR_BGR2GRAY);

            vector<Point> points =
                    textureDescriptor->describePoints(
                            grayImage,
                            stencil,
                            foregroundDescriptors.rowRange(
                                    sampleInfos.size(),
                                    sampleInfos.size() + processingSize.area()));

            for (auto const& p : points)
            {
                sampleInfos.push_back({iFrame, p, mirrored});
            }
	    }
	}

	Mat descriptors = foregroundDescriptors.rowRange(0,sampleInfos.size()).reshape(1);

	normalizer->train(descriptors);
	normalizer->apply(descriptors, descriptors);


	Mat textonLabels;
	cv::kmeans(
	        descriptors,
			nTextons,
			textonLabels,
	        TermCriteria{CV_TERMCRIT_EPS+CV_TERMCRIT_ITER, 100, 0.1},
	        4,
			KMEANS_PP_CENTERS,
			textonCenters);


	//illustrateTextonMaps(sampleInfos, textonLabels, images);
	//savePrototypicalTextonInstances(foregroundDescriptors, sampleInfos, textonCenters, images);
}

auto TextonCreator::
getTextonMap(Mat3b const& imageFull) -> cv::Mat1b
{
	Mat3b image = cvret::resize(imageFull, processingSize);
	Mat1b grayImage = cvret::cvtColor(image, COLOR_BGR2GRAY);

	Mat1f features{processingSize.area(), textureDescriptor->getDescriptorSize()};

	vector<Point> points =
	        textureDescriptor->describePoints(grayImage, cv::noArray(), features);

	normalizer->apply(features, features);

	cv::Mat1b textonMap{processingSize,0};

	int iSample = 0;
	for (Point p : points)
	{
		auto iTextons = cvx::irange(textonCenters.rows);
		textonMap(p) =
            *stdx::min_element_by(
                iTextons.begin(), iTextons.end(),
                [&](int iTexton)
                {
                    return cvx::FrobeniusSq(textonCenters.row(iTexton)-features.row(iSample));
                });

		++iSample;
	}

	return textonMap;
}

void TextonCreator::
savePrototypicalTextonInstances(
        Mat1f const& foregroundDescriptors,
        vector<SampleInfo> const& sampleInfos,
        Mat1f const& textonCenters,
        ImageLoadingIterable const& images) const
{
    for (int iTexton : cvx::irange(textonCenters.rows))
    {
        top_queue<int> queue(10);
        for (int iSample : cvx::irange(foregroundDescriptors.rows))
        {
            double dist = cv::norm(foregroundDescriptors.row(iSample) - textonCenters.row(iTexton));
            queue.push(iSample, -dist);
        }

        int rank = 0;
        for (int iSample : queue.popAll())
        {
            auto sampleInfo = sampleInfos[iSample];

            Vec2i wing{16,16};

            auto img = (sampleInfo.mirrored==1 ? cvret::flip(images[sampleInfo.iFrame], 1) : images[sampleInfo.iFrame]);
            Point bigP = sampleInfo.p*img.size()/processingSize;
            Vec2i bigWing = wing*img.size()/processingSize;

            auto bordered = cvret::copyMakeBorder(img, bigWing[0], bigWing[0], bigWing[0], bigWing[0], BORDER_CONSTANT, Scalar(0));
            Mat part = bordered(Rect{bigP, Size{bigWing[0]*2+1, bigWing[0]*2+1}});

            cvx::imwrite(config::RESULT_PATH/"textons"/cvx::str::format("texton%d_rank%d.jpg", iTexton, rank), part);
            ++rank;
        }
    }
}

void TextonCreator::
illustrateTextonMaps(
        vector<SampleInfo> const& sampleInfos,
        Mat1b const& textonLabels,
        ImageLoadingIterable const& images) const
{
    pyx::Pyplot plt;
    auto fig_axarr = plt.subplots({1,nTextons}, "sharey=True,sharex=True,figsize=(8,4)");
    auto fig = fig_axarr[0];
    auto axarr = fig_axarr[1];

    for (int iTexton : cvx::irange(nTextons))
    {
        axarr[iTexton].call("stem", {textonCenters.row(iTexton)});
    }
    plt.showAndClose();

    Size displaySize = Size{960,540}/2;

    int iSample = 0;
    int prevIFrame = -1;

    Mat1b textonMap{processingSize};

    for (auto const& sampleInfo : sampleInfos)
    {
        if (sampleInfo.iFrame != prevIFrame)
        {
            if (prevIFrame > -1)
            {
                Mat3b image = images[sampleInfo.iFrame];
                cvx::resizeBest(image, image, displaySize);

                Mat3b textonNice = illustrateLabels(textonMap, nTextons);
                cv::resize(textonNice, textonNice, displaySize, 0,0, INTER_NEAREST);

                Mat1b textonMapResized = cvret::resize(textonMap, displaySize, 0,0, INTER_NEAREST);

                cv::imshow("blend", cvret::addWeighted(textonNice, 0.5, image, 0.5, 0.0));

                for (int i : cvx::irange(nTextons))
                {
                    cv::imshow(std::to_string(i), cvx::visu::maskIllustration(image, textonMapResized==i));
                }

                cv::imshow("textons", textonNice);
                cv::imshow("image", image);
                cv::waitKey();
            }

            textonMap = (uchar)nTextons;
        }

        textonMap(sampleInfo.p) = textonLabels.at<uchar>(iSample);

        ++iSample;
        prevIFrame = sampleInfo.iFrame;
    }
}


auto TextonCreator::
describe() const -> boost::property_tree::ptree
{
    boost::property_tree::ptree pt;

    pt.put("nTextons", nTextons);
    pt.put("processingSize_w", processingSize.width);
    pt.put("processingSize_h", processingSize.height);
    pt.put("mirroredAlso", mirroredAlso);
    pt.put_child("textureDescriptor", textureDescriptor->describe());

    if (normalizer)
    {
        pt.put_child("normalizer", normalizer->describe());
    }

    return pt;
}

auto TextonCreator::
create(boost::property_tree::ptree const& pt) -> std::unique_ptr<TextonCreator>
{
    if (pt.count("normalizer")==0)
    {
        return stdx::make_unique<TextonCreator>(
                Size{pt.get<int>("processingSize_w"), pt.get<int>("processingSize_h")},
                pt.get<int>("nTextons"),
                *TextureDescriptor::create(pt.get_child("textureDescriptor")),
                pt.get<bool>("mirroredAlso")
                );
    }

    return stdx::make_unique<TextonCreator>(
            Size{pt.get<int>("processingSize_w"), pt.get<int>("processingSize_h")},
            pt.get<int>("nTextons"),
            *TextureDescriptor::create(pt.get_child("textureDescriptor")),
            *DataNormalizer::create(pt.get_child("normalizer")),
            pt.get<bool>("mirroredAlso")
            );

}

