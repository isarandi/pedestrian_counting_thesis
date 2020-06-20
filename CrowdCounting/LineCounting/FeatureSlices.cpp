#include <boost/filesystem.hpp>
#include <cvextra/core.hpp>
#include <cvextra/cvenums.hpp>
#include <cvextra/cvret.hpp>
#include <cvextra/improc.hpp>
#include <cvextra/ImageLoadingIterable.hpp>
#include <cvextra/math.hpp>
#include <cvextra/strings.hpp>
#include <cvextra/volumetric.hpp>
#include <Illustrate/illustrate.hpp>
#include <CrowdCounting/LineCounting/FeatureSlices.hpp>
#include <CrowdCounting/LineCounting/Features/Textons/TextonCreator.hpp>
#include <CrowdCounting/RegionCounting/Features/Extractors/FilterBank.hpp>
#include <Flow/flowSlice.hpp>
#include <Flow/lineFlow.hpp>
#include <Run/config.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/core/mat.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <Persistence.hpp>
#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/json_parser.hpp>
#include <string>

using namespace std;
using namespace cv;
using namespace cvx;
using namespace crowd;
using namespace crowd::lineopticalflow;
using namespace crowd::linecounting;

auto FeatureSlices::
operator () (
        cv::Rect const& rect
) const -> FeatureSlices
{
    return {
        image(rect),
                flow(rect),
                directionLabels(rect),
                canny(rect),
                grad(rect),
                foregroundMask(rect),
                stencil(rect),
                textonMap(rect),
                mirroredTextonMap(rect),
                scaleMap(rect)};
}

auto FeatureSlices::
operator () (
        cv::Range const& rowRange,
        cv::Range const& colRange
) const -> FeatureSlices
{
    return {
        image(rowRange, colRange),
                flow(rowRange, colRange),
                directionLabels(rowRange, colRange),
                canny(rowRange, colRange),
                grad(rowRange, colRange),
                foregroundMask(rowRange, colRange),
                stencil(rowRange, colRange),
                textonMap(rowRange, colRange),
                mirroredTextonMap(rowRange, colRange),
                scaleMap(rowRange, colRange),
                nTextons};
}

auto crowd::
createCannySlice(
        cvx::ImageLoadingIterable const& images,
        cvx::LineSegment const& seg,
        CannyOptions const& options
) -> BinaryMat
{
    BinaryMat cannySlice{(int)images.size(), seg.floorLength()};
    auto dilateStructElem = cv::getStructuringElement(MORPH_RECT, {options.dilateSize, 1});

    int iFrame = 0;
    for (Mat image : images)
    {
        auto grayImage = cvret::cvtColor(image, COLOR_BGR2GRAY);
        auto edges = cvxret::cannyPlus(grayImage, options.gaussRadius, options.threshold1, options.threshold2, options.minEdgeLength);
        cv::dilate(edges, edges, dilateStructElem);

        Mat targetRow = cannySlice.row(iFrame++);
        cvx::lineProfile(edges, targetRow, seg);
    }

    return cannySlice;
}

auto crowd::
createGradientSlice(
        cvx::ImageLoadingIterable const& images,
        cvx::LineSegment const& seg,
        int dilateSize
) -> Mat2d
{
    Mat2d gradSlice{(int)images.size(), seg.floorLength()};
    auto dilateStructElem = cv::getStructuringElement(MORPH_RECT, {dilateSize, 1});

    int iFrame = 0;
    for (Mat image : images)
    {
        auto grayImage = cvret::cvtColor(image, COLOR_BGR2GRAY);
        Mat2f grad = cvxret::gradientField(grayImage);
        cv::dilate(grad, grad, dilateStructElem);

        Mat profile = cvxret::lineProfile(grad, seg);
        Mat targetRow = gradSlice.row(iFrame);
        profile.copyTo(targetRow);

        ++iFrame;
    }

    return gradSlice;
}

auto FeatureSlices::
horizontalFlipped() const -> FeatureSlices
{
    auto newFlow = flow.clone();
    auto newGrad = grad.clone();
    auto newDirectionLabels = directionLabels.clone();
    cvx::setChannel(-cvret::extractChannel(newFlow,0), newFlow, 0);
    cvx::setChannel(-cvret::extractChannel(newGrad,0), newGrad, 0);

    newDirectionLabels.setTo(1, directionLabels == 2);
    newDirectionLabels.setTo(2, directionLabels == 1);

    return FeatureSlices{image, newFlow, newDirectionLabels, canny, newGrad, foregroundMask, stencil, mirroredTextonMap, textonMap, scaleMap, nTextons};
}

void crowd::CannyOptions::
writeToFile(
        cvx::bpath const& path)
{
    boost::property_tree::ptree pt;
    pt.put("gaussRadius", gaussRadius);
    pt.put("threshold1", threshold1);
    pt.put("threshold2", threshold2);
    pt.put("dilateSize", dilateSize);
    pt.put("minEdgeLength", minEdgeLength);
    boost::filesystem::create_directories(path.parent_path());
    boost::property_tree::write_json(path.string(), pt);
}

auto crowd::CannyOptions::
fromFile(cvx::bpath const& path) -> CannyOptions
{
    boost::property_tree::ptree pt;
    boost::property_tree::read_json(path.string(), pt);

    return CannyOptions{
        pt.get<double>("gaussRadius"),
                pt.get<double>("threshold1"),
                pt.get<double>("threshold2"),
                pt.get<int>("dilateSize"),
                pt.get<int>("minEdgeLength")
    };
}

auto crowd::FeatureSlices::
loadOrCompute(
        std::string const& datasetName,
        cvx::LineSegment const& segment) -> FeatureSlices
{
    string name =
            cvx::str::format(
                    "%s, line_segment {{%.5f,%.5f},{%.5f,%.5f}}",
                    datasetName,
                    segment.p1.x,
                    segment.p1.y,
                    segment.p2.x,
                    segment.p2.y);

    FeatureSlices slices;
    auto images = config::frames(datasetName);

    slices.image = Persistence::loadOrComputeMat(name+"image", [&](){
        return cvx::timeSlice(images, segment);
    });

    BinaryMat frameStencil = config::roiStencil(datasetName);
    BinaryMat stencilRow = cvxret::lineProfile(frameStencil, segment * (frameStencil.rows/(double)images[0].rows));
    slices.stencil = cv::repeat(stencilRow, slices.image.rows, 1);

    slices.flow = Persistence::loadOrComputeMat(name+"flow", [&]()
    {
        OpticalFlowOptions flowOptions =
                OpticalFlowOptions::fromFile(config::configPath(datasetName, "flow"));
        return crowd::lineopticalflow::createFlowSliceParallel(images, segment, flowOptions);
    });
    // set infinties and NaNs to 0
    slices.flow.setTo(Scalar::all(0), Mat((cvret::magnitude(slices.flow)<1000)==0));

    slices.canny = Persistence::loadOrComputeMat(name+"canny", [&]()
    {
        CannyOptions options = CannyOptions::fromFile(config::configPath(datasetName, "canny"));
        auto dilateStructElem = cv::getStructuringElement(MORPH_RECT, {options.dilateSize, 1});

        if (config::dilatedCannys(datasetName).size()==0)
        {
            int i=0;
            for (Mat3b image : config::frames(datasetName))
            {
                auto grayImage = cvret::cvtColor(image, COLOR_BGR2GRAY);
                auto edges = cvxret::cannyPlus(grayImage, options.gaussRadius, options.threshold1, options.threshold2, options.minEdgeLength);
                cv::dilate(edges, edges, dilateStructElem);
                cvx::imwrite(config::dilatedCannyPath(datasetName)/cvx::str::format("%06d.png", i), edges);
                ++i;
            }
        }
        BinaryMat cannySlice = cvx::timeSlice(config::dilatedCannys(datasetName), segment);
        return cannySlice;
    });

    slices.grad = Mat2d::zeros(slices.image.size());

//    Persistence::loadOrComputeMat(name+"grad", [&](){
//        return crowd::createGradientSlice(images, segment, 10);
//    });

   // cout << datasetName << "_grad" << endl;

    slices.foregroundMask = Persistence::loadOrComputeMat(name+"foregroundMask", [&]()
    {
        auto foregroundMasks = config::masks(datasetName);
        BinaryMat fgSlice = cvx::timeSlice(foregroundMasks, segment * (foregroundMasks[0].rows/(double)images[0].rows));
        cv::resize(fgSlice, fgSlice, Size{segment.floorLength(), fgSlice.rows}, 0,0, INTER_NEAREST);

        fgSlice.setTo(0, slices.stencil==0);
        return fgSlice;
    });

    slices.loadOrComputeTextons(datasetName, name, segment);

    Mat1d scaleMap = config::scaleMap(datasetName);
    Mat1d scaleRow = cvxret::lineProfile(scaleMap, segment * (frameStencil.rows/(double)images[0].rows));
    slices.scaleMap = cv::repeat(scaleRow, slices.image.rows, 1);
    slices.directionLabels = Mat1b::zeros(slices.size());

    return slices;
}

void FeatureSlices::
loadOrComputeTextons(
        const std::string& datasetName,
        const std::string& fullName,
        const cvx::LineSegment& segment)
{
    Size frameSize = config::frames(datasetName)[0].size();


    nTextons = 6;
    textonMap = Persistence::loadOrComputeMat(fullName+"textonMap/6/gray/LM49"+std::to_string(nTextons), [&]()
    {
        cout << "inside" << endl;
        TextonCreator tc {
            Size{960,540}/8,
            nTextons,
            FilterBankDescriptor{49},
            PerFeatureNormalizer{},
            false};

        tc.train(
                cvx::imagesIn({
                        config::framePath("crange_densities/12"),
                        config::framePath("crange_densities/13"),
                        config::framePath("crange_densities/21"),
                        config::framePath("crange_densities/23"),
                        config::framePath("crange_densities/24"),
                        config::framePath("crange_densities/31"),
                        config::framePath("crange_densities/32"),
                        config::framePath("crange_densities/33"),
                        config::framePath("crange_densities/41"),
                        config::framePath("crange_densities/42"),
                        config::framePath("crange_densities/44")}),
                config::roiStencil("crange_ausschnitt1"),
                "densities");

//        tc.train(
//                cvx::imagesIn(
//                        config::framePath("ucsd_vidd1_orig")).range(0,2000),
//                config::roiStencil("crange_ausschnitt1"),
//                "ucsd_vidd1_orig");

        if (config::textons(datasetName).size()==0)
        {
            int i=0;
            for (Mat3b image : config::frames(datasetName))
            {
                cvx::imwrite(config::textonPath(datasetName)/cvx::str::format("%06d.png", i), tc.getTextonMap(image));
                ++i;
            }
        }

        cout << "inside" << endl;
        Mat1b textonSlice =
                cvx::timeSlice(
                        config::textons(datasetName),
                        segment * (tc.processingSize.height/(double)frameSize.height));

        cv::resize(textonSlice, textonSlice, Size{segment.floorLength(), textonSlice.rows}, 0,0, INTER_NEAREST);
        return textonSlice;
    });


    mirroredTextonMap = Persistence::loadOrComputeMat(fullName+"mirroredTextonMap/6/gray/LM49"+std::to_string(nTextons), [&]()
    {
        TextonCreator tc{
            Size{960,540}/8,
            nTextons,
            FilterBankDescriptor{49},
            PerFeatureNormalizer{},
            false};

        tc.train(
                cvx::imagesIn({
                        config::framePath("crange_densities/12"),
                        config::framePath("crange_densities/13"),
                        config::framePath("crange_densities/21"),
                        config::framePath("crange_densities/23"),
                        config::framePath("crange_densities/24"),
                        config::framePath("crange_densities/31"),
                        config::framePath("crange_densities/32"),
                        config::framePath("crange_densities/33"),
                        config::framePath("crange_densities/41"),
                        config::framePath("crange_densities/42"),
                        config::framePath("crange_densities/44")}),
                config::roiStencil("crange_ausschnitt1"),
                "densities");

//        tc.train(
//                cvx::imagesIn(
//                        config::framePath("ucsd_vidd1_orig")).range(0,2000),
//                config::roiStencil("crange_ausschnitt1"),
//                "ucsd_vidd1_orig");

        if (config::textons(datasetName+"_mirrored").size()==0)
        {
            int i=0;
            for (Mat3b image : config::frames(datasetName))
            {
                Mat1b textonMap = cvret::flip(tc.getTextonMap(cvret::flip(image, 1)), 1);
                cvx::imwrite(config::textonPath(datasetName+"_mirrored")/cvx::str::format("%06d.png", i), textonMap);
                ++i;
            }
        }

        Mat1b textonSlice =
                cvx::timeSlice(
                        config::textons(datasetName+"_mirrored"),
                        segment * (tc.processingSize.height/(double)frameSize.height));

        cv::resize(textonSlice, textonSlice, Size{segment.floorLength(), textonSlice.rows}, 0,0, INTER_NEAREST);
        return textonSlice;
    });
}
