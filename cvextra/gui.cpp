#include "gui.hpp"
#include <utility>
#include <map>
#include <memory>
#include <functional>

#include <boost/uuid/uuid.hpp>            // uuid class
#include <boost/uuid/uuid_generators.hpp> // generators
#include <boost/lexical_cast.hpp>

#include <cvextra.hpp>

using namespace std;
using namespace cv;
using namespace cvx;
using namespace cvx::gui;
using boost::uuids::uuid;

typedef std::function<void(int, cv::Point, int)> MouseEventHandler;

static map<string, unique_ptr<map<uuid, MouseEventHandler>>> mouseHandlers;
static boost::uuids::random_generator uuidGen;

static void mouseCallbackTrampoline(int event, int x, int y, int flags, void* userdata)
{
    auto& handlersWithID = *static_cast<map<uuid, MouseEventHandler>*>(userdata);

    for (auto& pair : handlersWithID)
    {
        MouseEventHandler& handler = pair.second;
        handler(event, Point(x,y), flags);
    }
}

auto cvx::gui::
addMouseEventHandler(string const& winname, MouseEventHandler handler) -> uuid
{
    uuid u = uuidGen();

    if (mouseHandlers.count(winname) == 0)
    {
        auto up_map = stdx::make_unique<map<uuid, MouseEventHandler>>();
        up_map->insert({u, std::move(handler)});

        cv::setMouseCallback(winname, mouseCallbackTrampoline, up_map.get());
        mouseHandlers[winname] = std::move(up_map);

    } else
    {
        mouseHandlers[winname]->insert({u, std::move(handler)});
    }

    return u;
}

void cvx::gui::
removeMouseEventHandler(string const& winname, uuid const& handlerID)
{
    if (mouseHandlers.count(winname))
    {
        mouseHandlers[winname]->erase(handlerID);

        if (mouseHandlers[winname]->empty())
        {
            mouseHandlers.erase(winname);
            cv::setMouseCallback(winname, nullptr, nullptr);
        }
    }
}

void cvx::gui::
removeMouseEventHandlers(string const& winname)
{
    mouseHandlers.erase(winname);
    cv::setMouseCallback(winname, nullptr, nullptr);
}


//typedef std::function<void(bool)> ButtonEventHandler;
//
//static void buttonCallbackTrampoline(int event, void* userdata)
//{
//    auto& handler = *static_cast<ButtonEventHandler*>(userdata);
//    handler(event == 1);
//}
//
//static vector<unique_ptr<ButtonEventHandler>> buttonHandlers;

//void cvx::gui::createButton(string const& name, ButtonEventHandler handler, int buttonType, bool initialState)
//{
//    auto up_handler = stdx::make_unique<ButtonEventHandler>(handler);

//    cv::createButton(name, buttonCallbackTrampoline, up_handler.get(), buttonType, initialState);
//    buttonHandlers.push_back(std::move(up_handler));
//}

typedef std::function<void(int)> TrackbarEventHandler;
static map<string, vector<unique_ptr<TrackbarEventHandler>>> trackbarHandlers;
static map<string, vector<unique_ptr<int>>> trackbarVariables;

static void trackbarCallbackTrampoline(int pos, void* userdata)
{
    auto& handler = *static_cast<TrackbarEventHandler*>(userdata);
    handler(pos);
}

void cvx::gui::
createTrackbar(
        string const& winname,
        string const& trackbarName,
        int initialPos,
        int count,
        TrackbarEventHandler handler)
{
    auto up_handler = stdx::make_unique<TrackbarEventHandler>(handler);
    auto up_variable = stdx::make_unique<int>(initialPos);

    cv::createTrackbar(trackbarName, winname, up_variable.get(), count, trackbarCallbackTrampoline, up_handler.get());

    trackbarHandlers[winname].emplace_back(std::move(up_handler));
    trackbarVariables[winname].emplace_back(std::move(up_variable));
}


void cvx::gui::
createTrackbar(
        const string& winname,
        const string& trackbarName,
        double initValue,
        double min,
        double max,
        function<void (double)> handler)
{
    int nTrackbarSteps = 1000;

    auto up_handler = stdx::make_unique<TrackbarEventHandler>([=](int pos){
        double param = cvx::math::linearRescale(pos, 0, nTrackbarSteps, min, max);
        handler(param);
    });

    auto up_variable =
            stdx::make_unique<int>(
                static_cast<int>(
                    cvx::math::linearRescale(initValue, min, max, 0, 1000)));

    cv::createTrackbar(trackbarName, winname, up_variable.get(), nTrackbarSteps, trackbarCallbackTrampoline, up_handler.get());

    trackbarHandlers[winname].emplace_back(std::move(up_handler));
    trackbarVariables[winname].emplace_back(std::move(up_variable));
}


void TweakableDisplay::
show()
{
    cv::namedWindow(this->name);

    for (Parameter p: params)
    {
        cvx::gui::createTrackbar(
                    this->name,
                    p.name,
                    p.defaultValue,
                    p.minValue,
                    p.maxValue,
                    [=](double val)
        {
            this->currentParams.at(p.name) = val;

            stringstream ss;
            for (Parameter paramToWrite: params)
            {
                ss << paramToWrite.name << " = " << this->currentParams.at(paramToWrite.name) << ";" << endl;
            }
            ss << "---------";
            cout << ss.str() << endl;

            if (!this->updateOnClick)
            {
            	showCurrent();
            }
        });
    }

    if (updateOnClick)
    {
        cvx::gui::addMouseEventHandler(this->name, [=](int event, Point, int){
            if (event == EVENT_LBUTTONDOWN)
            {
            	showCurrent();
            }
        });
    }

    showCurrent();
}

void TweakableDisplay::
showCurrent()
{
	cv::imshow(this->name, this->imageUpdater(this->currentParams));
}

void cvx::gui::
startTweaking(
		std::vector<TweakableDisplay::Parameter> const& params,
		std::function<cv::Mat(std::map<std::string, double> const&)> imageUpdater,
		bool updateOnClick)
{
	auto display = TweakableDisplay(updateOnClick, boost::filesystem::unique_path("%%%%%").string(), params, imageUpdater);
	display.show();
	cvx::waitKey(' ');
}
