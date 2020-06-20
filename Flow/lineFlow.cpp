#include "lineFlow.hpp"
#include "Run/config.hpp"
#include "featuresForFlow.hpp"
#include "Illustrate/illustrate.hpp"
#include <cvextra.hpp>
#include <boost/filesystem.hpp>
#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/json_parser.hpp>
#include <iostream>
#include <string>
#include <queue>
#include <omp.h>

using namespace std;
using namespace cv;
using namespace cvx;
using namespace cvx::math;

using namespace crowd;
using namespace crowd::lineopticalflow::details;

auto crowd::lineopticalflow::details::
calcEnergy(
        vector<vector<Mat1d>> const& images,
        vector<double> const& channelWeights,
        int iTargetImage,
        Mat2d const& flow,
        OpticalFlowOptions const& options
        ) -> double
{
    Size frameSize = images[iTargetImage][0].size();
    int nLineSamples = flow.cols;
    int horizontalMidpoint = (frameSize.width-1)/2;

    int nChannels = channelWeights.size();

    double totalEnergy = 0;
    int nFlowVectorsPointingOutsideFrame = 0;

    for (int iSample : cvx::irange(nLineSamples))
    {
        Point p{horizontalMidpoint, iSample};
        Vec2d flowHere = flow(iSample);

        //--- Calculate the data term
        double EdataSq = 0;

        // Go through the nearby frames
        for (int iOtherImage : cvx::irange(images.size()))
        {
            if (iTargetImage == iOtherImage)
            {
                continue;
            }

            double timeDistance = iOtherImage-iTargetImage;
            double timeWeight = std::exp(-0.35 * std::abs(timeDistance));
            Point2d pointInOtherImage = p + timeDistance * flowHere;

            if (cvx::contains(frameSize, pointInOtherImage))
            {
                for (int iChannel : cvx::irange(nChannels))
                {
                    double targetImageValue = images[iTargetImage][iChannel](p);
                    double otherImageValue = cvx::bilinear<double>(images[iOtherImage][iChannel], pointInOtherImage);

                    EdataSq += timeWeight * channelWeights[iChannel] * cvx::sq(otherImageValue - targetImageValue);
                }
            } else
            {
                ++nFlowVectorsPointingOutsideFrame;
            }
        }
        //---

        //--- Calculate the smoothness term
        Vec2d uvDerivative = (iSample + 1 < nLineSamples && iSample >=0)
                ? 0.5*(flow(iSample+1) - flow(iSample-1)) : 0;
        double EsmoothSq = cvx::FrobeniusSq(uvDerivative);
        //---

        //--- Calculate the magnitude term
        double EmagnitudeSq = cvx::FrobeniusSq(flowHere);
        //---

        totalEnergy += std::sqrt(EdataSq + options.epsilonSquare)
                + options.smoothnessAlpha * std::sqrt(EsmoothSq + options.epsilonSquare)
                + options.magnitudeBeta  * std::sqrt(EmagnitudeSq + options.epsilonSquare);
    }

    cout << " (flow vectors pointing outside: " << nFlowVectorsPointingOutsideFrame << ") ";

    return totalEnergy/frameSize.area();
}

void crowd::lineopticalflow::details::
calcPsiData(
        vector<vector<Mat1d>> const& images,
        vector<double> const& channelWeights,
        int iTargetImage,
        Mat2d const& flowPlusDeltaFlow,
        Mat1d& psiDataOut,
        double epsilonSq)
{
    Size frameSize = images[iTargetImage][0].size();
    int horizontalMidpoint = (frameSize.width-1)/2;

    for (int iSample : cvx::irange(flowPlusDeltaFlow.cols))
    {
        Point p{horizontalMidpoint, iSample};
        double EdataSq = 0;

        int nChannels = channelWeights.size();
        Vec2d flowHere = flowPlusDeltaFlow(iSample);

        // Go through the nearby frames
        for (int iOtherImage : cvx::irange(images.size()))
        {
            if (iTargetImage == iOtherImage)
            {
                continue;
            }

            double timeDistance = iOtherImage-iTargetImage;
            Point2d pointInOtherImage = p + timeDistance * flowHere;

            if (cvx::contains(frameSize, pointInOtherImage))
            {
                double timeWeight = std::exp(-0.35 * std::abs(timeDistance));
                for (int iChannel : cvx::irange(nChannels))
                {
                    double targetImageValue = images[iTargetImage][iChannel](p);
                    double otherImageValue = cvx::bilinear<double>(images[iOtherImage][iChannel], pointInOtherImage);

                    EdataSq += timeWeight * channelWeights[iChannel] * cvx::sq(otherImageValue - targetImageValue);
                }
            }
        }

        psiDataOut(iSample) = std::pow(EdataSq + epsilonSq, -0.5);
    }
}

void crowd::lineopticalflow::details::
calcPsiMagnitude(
        Mat2d const& flowPlusDeltaFlow,
        Mat1d& psiMagnitudeOut,
        double epsilonSq)
{
    for (int iSample : cvx::irange(flowPlusDeltaFlow.cols))
    {
        Vec2d uv = flowPlusDeltaFlow(iSample);
        double magnitudeSq = uv.dot(uv);
        psiMagnitudeOut(iSample) = std::pow(magnitudeSq + epsilonSq, -0.5);
    }
}

void crowd::lineopticalflow::details::
calcPsiSmooth(
        Mat2d const& flowPlusDeltaFlow,
        Mat1d& psiSmoothOut,
        double epsilonSq)
{
    int nSamples = flowPlusDeltaFlow.cols;
    for (int iSample : cvx::irange(nSamples))
    {
        Vec2d uvDerivative =
                (iSample + 1 < nSamples) ? flowPlusDeltaFlow(iSample+1) - flowPlusDeltaFlow(iSample) : 0;

        double EsmoothSq = cvx::FrobeniusSq(uvDerivative);
        psiSmoothOut(iSample) = std::pow(EsmoothSq + epsilonSq, -0.5);
    }
}

// Linearize the data term and fix values for Ix and Idiff so that the remaining problem is convex
void crowd::lineopticalflow::details::
linearizeDataTerm(
        vector<vector<Mat1d>> const& images,
        vector<double> const& channelWeights,
        int iTargetImage,
        vector<vector<Mat2d>> const& gradientFields,
        Mat2d const& flow,
        Mat2d& IxxOut,
        Mat1d& Ix1x2Out,
        Mat2d& IxdiffOut)
{
    Size frameSize = images[iTargetImage][0].size();
    int horizontalMidpoint = (frameSize.width-1)/2;
    int nChannels = channelWeights.size();

    // Matrix operations! Initialize to zero
    IxxOut.setTo(Vec2d{0.0,0.0});
    Ix1x2Out.setTo(0.0);
    IxdiffOut.setTo(Vec2d{0.0,0.0});
    //---

    for (int iSample : cvx::irange(flow.cols))
    {
        Point p{horizontalMidpoint, iSample};

        double weightedSumOfSquaredDifferences = 0;
        Vec2d weightedSumOfDifferenceTimesGradient = {0,0};

        for (int iOtherImage : cvx::irange(images.size()))
        {
            if (iTargetImage == iOtherImage)
            {
                continue;
            }

            double timeDistance = iOtherImage-iTargetImage;
            Point2d pointInOtherImage = p + timeDistance * flow(iSample);

            if (cvx::contains(frameSize, pointInOtherImage))
            {
                double timeWeight = std::exp(-0.35 * std::abs(timeDistance));
                for (int iChannel : cvx::irange(nChannels))
                {
                    double targetImageValue = images[iTargetImage][iChannel](p);

                    Vec2d otherImageGradient = cvx::bilinear<double,2>(gradientFields[iOtherImage][iChannel], pointInOtherImage);
                    double otherImageValue = cvx::bilinear<double>(images[iOtherImage][iChannel], pointInOtherImage);

                    double difference = otherImageValue - targetImageValue;
                    double weight = channelWeights[iChannel] * timeWeight;

                    weightedSumOfDifferenceTimesGradient += weight * difference * (timeDistance * otherImageGradient);
                    weightedSumOfSquaredDifferences += weight * cvx::sq(difference);
                }
            }
        }

        Vec2d w = weightedSumOfDifferenceTimesGradient;
        if (weightedSumOfSquaredDifferences != 0)
        {
            IxxOut(iSample)    += w.mul(w) / weightedSumOfSquaredDifferences;
            Ix1x2Out(iSample)  += w[0] * w[1] / weightedSumOfSquaredDifferences;
        }
        IxdiffOut(iSample) += w;
    }
}

void crowd::lineopticalflow::details::
successiveOverRelaxation(
        Mat2d const& flow,
        Mat2d& deltaFlow,
        Mat2d const& Ixx,
        Mat1d const& Ix1x2,
        Mat2d const& Ixdiff,
        Mat1d const& psiData,
        Mat1d const& psiSmooth,
        Mat1d const& psiMagnitude,
        OpticalFlowOptions const& options)
{
    for (int iSOR : cvx::irange(options.nIterationsSOR))
    {
        for (int iSample : cvx::irange(flow.cols))
        {
            double sumOfNeighborPsiSmooths = psiSmooth(iSample) + (iSample > 0 ? psiSmooth(iSample-1) : 0);

            for (int uORv : {0,1}) // indicator: if 0 then deltaU is being determined, otherwise deltaV
            {
                double flowHere = flow(iSample)[uORv];

                // Naming convention: Aii*xi + di = 0; where xi is the deltaFlow component
                double Aii =
                        psiData(iSample) * Ixx(iSample)[uORv]
                        + options.smoothnessAlpha * sumOfNeighborPsiSmooths
                        + options.magnitudeBeta * psiMagnitude(iSample);

                double di =
                        psiData(iSample) * (Ixdiff(iSample)[uORv] + Ix1x2(iSample) * deltaFlow(iSample)[1-uORv])
                        - options.smoothnessAlpha *
                          (
                            (iSample-1 >= 0      ? psiSmooth(iSample-1) * (flow(iSample-1)[uORv] + deltaFlow(iSample-1)[uORv] - flowHere) : 0) +
                            (iSample < flow.cols ? psiSmooth(iSample  ) * (flow(iSample  )[uORv] + deltaFlow(iSample  )[uORv] - flowHere) : 0)
                          )
                        + options.magnitudeBeta * psiMagnitude(iSample) * flow(iSample)[uORv];

                deltaFlow(iSample)[uORv] -= options.SORomega * (deltaFlow(iSample)[uORv] + di/Aii);
            }
        }

    }
}

auto crowd::lineopticalflow::details::
solveConvexProblem(
        vector<vector<Mat1d>> const& images,
        vector<double> const& channelWeights,
        int iTargetImage,
        Mat2d const& flow,
        Mat2d const& Ixx,
        Mat1d const& Ix1x2,
        Mat2d const& Ixdiff,
        OpticalFlowOptions const& options
        ) -> Mat2d
{
    int nLineSamples = flow.cols;

    Mat1d psiData{1,nLineSamples};          // psi_d
    Mat1d psiSmooth{1,nLineSamples};        // psi_s(p+1/2)
    Mat1d psiMagnitude{1,nLineSamples};     // psi_m

    Mat2d deltaFlow = Mat2d::zeros(1,nLineSamples);

    for (int l : cvx::irange(options.nIterationsL))
    {
        //--- Calculate fixed values for psiData and diffusivities so that the remaining problem is linear
        Mat2d flowPlusDeltaFlow = flow + deltaFlow;

        calcPsiData(images, channelWeights, iTargetImage, flowPlusDeltaFlow, psiData, options.epsilonSquare);
        calcPsiSmooth(flowPlusDeltaFlow, psiSmooth, options.epsilonSquare);
        calcPsiMagnitude(flowPlusDeltaFlow, psiMagnitude, options.epsilonSquare);
        //---

        //--- Solve linear problem
        successiveOverRelaxation(
                    flow, deltaFlow,
                    Ixx, Ix1x2, Ixdiff,
                    psiData, psiSmooth, psiMagnitude,
                    options);
        //---

//        double energy = calcEnergy(images, channelWeights, iTargetImage, flowPlusDeltaFlow, options);
//        cout << energy << endl;

//        int midpoint = (nLineSamples-1)/2;
//        int height = images[0][0].size().height;
//        Mat m = cvx::illust::inspectFlow(flowPlusDeltaFlow, Mat3b::zeros(images[0][0].size()),  Mat3b::zeros(images[0][0].size()), {{midpoint,height},{midpoint,height}})[0];
//        cvx::imwrite(config::DATA_PATH/"illustrations/crange_ausschnitt1/debug/scale"/cvx::str::format("l - %d, sor - %d.png", l, iSOR), m);

    }

    return deltaFlow;
}

auto crowd::lineopticalflow::details::
calcGradientFields(
        vector<vector<Mat1d>> const& images,
        int exceptIndex
        ) -> vector<vector<Mat2d> >
{
    vector<vector<Mat2d>> gradientFields;
    int index = 0;
    for (auto const& channels : images)
    {
        if (index != exceptIndex)
        {
            gradientFields.push_back(cvx::vectors::transform(channels, &cvxret::gradientField));
        } else
        {
            gradientFields.push_back({}); //placeholder
        }
        ++index;
    }
    return gradientFields;
}

auto crowd::lineopticalflow::details::
solveAtScale(
        vector<Mat> const& multiChannelImages,
        vector<double> const& channelWeights,
        int iTargetImage,
        Mat2d const& initialFlow,
        OpticalFlowOptions const& options
        ) -> Mat2d
{
    // Split each multi-channel image to a vector of single-channel images
    vector<vector<Mat1d>> images;
    for (auto const& im : multiChannelImages)
    {
        vector<Mat1d> channels;
        cv::split(im, channels);
        images.push_back(channels);
    }
    //---

    Mat2d flow = initialFlow.clone();

    // Declare the containers for various products of I_x_1, I_x_2 and I_diff
    Mat2d Ixx{flow.size()};   // I_x_1*I_x_1, I_x_2*I_x_2
    Mat1d Ix1x2{flow.size()}; // I_x_1*I_x_2
    Mat2d Ixdiff{flow.size()};// I_x_1*I_x_diff, I_x_1*I_x_diff
    //---

    // Calculate the gradient field of each channel of each image (except the targetImage)
    vector<vector<Mat2d>> gradientFields = calcGradientFields(images, iTargetImage); // gradientFields[iImage][iChannel] is a Mat2d field
    //---

    for (int k : cvx::irange(options.nIterationsK))
    {
        // Calculate Ixx, Ix1x2 and Ixdiff
        linearizeDataTerm(
                    images,
                    channelWeights,
                    iTargetImage,
                    gradientFields,
                    flow,
                    Ixx, Ix1x2, Ixdiff);
        //---

        Mat2d deltaFlow = solveConvexProblem(
                    images,
                    channelWeights,
                    iTargetImage,
                    flow,
                    Ixx, Ix1x2, Ixdiff,
                    options);

        flow += deltaFlow;
    }

    return flow;
}

auto crowd::lineopticalflow::
solveOpticalFlow(
        vector<Mat> const& images,
        vector<double> const& channelWeights,
        int iTargetImage,
        Mat2d const& initialFlow,
        OpticalFlowOptions const& options
        ) -> Mat2d
{
    // Create scaled versions of the images and the line segment
    vector<vector<Mat>> scaledImages = {images}; //scaledImages[iScale][iImage]

    Mat2d flow = initialFlow.clone();
    double fullHeight = images[0].rows;
    double scaledIdealHeight = fullHeight;

    // Go from fine to coarse: create pyramid and scale down the initial flow
    for (int iScale : cvx::irange(1,options.nScales))
    {
        scaledIdealHeight *= 0.5;
        Size downScaledSize = {(scaledImages.back()[0].cols-1)/2+1, (int)scaledIdealHeight};

        vector<Mat> imagesAtCoarserScale;
        for (auto const& imageAtPrevScale : scaledImages.back())
        {
            imagesAtCoarserScale.push_back(cvret::pyrDown(imageAtPrevScale, downScaledSize));
        }

        scaledImages.push_back(imagesAtCoarserScale);

        int nSamplesPrev = flow.cols;
        int nSamplesNow = downScaledSize.height;
        double scalingFactor = ((double)nSamplesNow)/nSamplesPrev;
        cv::pyrDown(flow, flow, {nSamplesNow,1});
        flow *= scalingFactor;
    }
    //---

    // Solve optical flow from the coarsest scale to finest
    for (int iScale = options.nScales-1; iScale >= 0; --iScale)
    {
        // Scale up the previous flow, except at the coarsest scale where we have nothing to scale up yet
        if (iScale != options.nScales-1)
        {
            int nSamplesNext = scaledImages[iScale][0].rows;
            cv::pyrUp(flow, flow, {nSamplesNext,1});
            flow *= 2.0;
        }

        flow = solveAtScale(
                    scaledImages[iScale],
                    channelWeights,
                    iTargetImage,
                    flow,
                    options);
    }
    //---

    return flow;
}


auto crowd::lineopticalflow::details::
getRotatedRectExtractorMatrix(RotatedRect rotatedRect) -> Mat
{
    Point2f rectPoints[4];
    rotatedRect.points(rectPoints);

    std::vector<Point2f> sourcePoints{rectPoints[0], rectPoints[1], rectPoints[2]};
    float h = rotatedRect.size.height;
    float w = rotatedRect.size.width;
    vector<Point2f> targetPoints{{0,h},{0,0},{w,0}};

    return cv::getAffineTransform(sourcePoints, targetPoints);
}

void crowd::lineopticalflow::OpticalFlowOptions::
writeToFile(
		cvx::bpath const& path)
{
	boost::property_tree::ptree pt;
	pt.put("smoothnessAlpha", smoothnessAlpha);
	pt.put("magnitudeBeta", magnitudeBeta);
	pt.put("epsilonSquare", epsilonSquare);
	pt.put("SORomega", SORomega);
	pt.put("nScales", nScales);
	pt.put("nIterationsK", nIterationsK);
	pt.put("nIterationsL", nIterationsL);
	pt.put("nIterationsSOR", nIterationsSOR);
	boost::filesystem::create_directories(path.parent_path());
	boost::property_tree::write_json(path.string(), pt);
}

auto crowd::lineopticalflow::OpticalFlowOptions::
fromFile(cvx::bpath const& path) -> OpticalFlowOptions
{
	boost::property_tree::ptree pt;
	boost::property_tree::read_json(path.string(), pt);

	return OpticalFlowOptions{
		pt.get<double>("smoothnessAlpha"),
		pt.get<double>("magnitudeBeta"),
		pt.get<double>("epsilonSquare"),
		pt.get<double>("SORomega"),
		pt.get<int>("nScales"),
		pt.get<int>("nIterationsK"),
		pt.get<int>("nIterationsL"),
		pt.get<int>("nIterationsSOR")
	};
}
