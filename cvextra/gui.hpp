#ifndef GUI_HPP
#define GUI_HPP

#include <opencv2/opencv.hpp>
#include <boost/uuid/uuid.hpp>

#include <memory>
#include <functional>
#include <string>
#include "LoopRange.hpp"

namespace cvx{ namespace gui {

auto addMouseEventHandler(
        std::string const& winname,
        std::function<void(int, cv::Point, int)> handler) -> boost::uuids::uuid;

void removeMouseEventHandler(std::string const& winname, boost::uuids::uuid const& handlerID);
void removeMouseEventHandlers(std::string const& winname);

//void createButton(
//        std::string const& name,
//        std::function<void(bool)> handler,
//        int buttonType = CV_PUSH_BUTTON,
//        bool initialState = false);

void createTrackbar(
        std::string const& winname,
        std::string const& trackbarName,
        int initialPos,
        int count,
        std::function<void(int)> handler);

void createTrackbar(
        std::string const& winname,
        std::string const& trackbarName,
        double initValue,
        double min,
        double max,
        std::function<void(double)> handler);



class TweakableDisplay
{
public:
    struct Parameter {
        std::string name;
        double minValue;
        double maxValue;
        double defaultValue;
    };

    TweakableDisplay(
            bool updateOnClick,
            std::string const& name,
            std::vector<Parameter> const& params,
            std::function<cv::Mat(std::map<std::string, double> const&)> imageUpdater)
        : updateOnClick(updateOnClick)
        , name(name)
        , params(params)
        , imageUpdater(imageUpdater)
    {
        for (Parameter p: params)
        {
            currentParams[p.name] = p.defaultValue;
        }
    }

    void show();

private:
    void showCurrent();


    bool updateOnClick;
    std::string name;
    std::vector<Parameter> params;
    std::map<std::string, double> currentParams;
    std::function<cv::Mat(std::map<std::string, double> const&)> imageUpdater;

};

void startTweaking(
		std::vector<TweakableDisplay::Parameter> const& params,
		std::function<cv::Mat(std::map<std::string, double> const&)> imageUpdater,
		bool updateOnClick = false);


}}
#endif // GUI_HPP
