#ifndef PYTHON_PYPLOT_HPP_
#define PYTHON_PYPLOT_HPP_

#include <Python/EasyCallable.hpp>
#include <Python/EasyObject.hpp>
#include <boost/filesystem.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

namespace pyx {

class DummyInit {
public:
    DummyInit(){
        pyx::initialize();
        pyx::import("matplotlib.pylab");
        pyx::import("io");
        pyx::run("import pickle");
    }
};

class Pyplot {
public:
    DummyInit d;
    EasyCallable const acorr;
    EasyCallable const angle_spectrum;
    EasyCallable const annotate;
    EasyCallable const arrow;
    EasyCallable const autoscale;
    EasyCallable const autumn;
    EasyCallable const axes;
    EasyCallable const axhline;
    EasyCallable const axhspan;
    EasyCallable const axis;
    EasyCallable const axvline;
    EasyCallable const axvspan;
    EasyCallable const bar;
    EasyCallable const barbs;
    EasyCallable const barh;
    EasyCallable const bone;
    EasyCallable const box;
    EasyCallable const boxplot;
    EasyCallable const broken_barh;
    EasyCallable const cla;
    EasyCallable const clabel;
    EasyCallable const clf;
    EasyCallable const clim;
    EasyCallable const close;
    EasyCallable const cohere;
    EasyCallable const colorbar;
    EasyCallable const colormaps;
    EasyCallable const colors;
    EasyCallable const connect;
    EasyCallable const contour;
    EasyCallable const contourf;
    EasyCallable const cool;
    EasyCallable const copper;
    EasyCallable const csd;
    EasyCallable const dedent;
    EasyCallable const delaxes;
    EasyCallable const disconnect;
    EasyCallable const draw;
    EasyCallable const draw_if_interactive;
    EasyCallable const errorbar;
    EasyCallable const eventplot;
    EasyCallable const figaspect;
    EasyCallable const figimage;
    EasyCallable const figlegend;
    EasyCallable const fignum_exists;
    EasyCallable const figtext;
    EasyCallable const figure;
    EasyCallable const fill;
    EasyCallable const fill_between;
    EasyCallable const fill_betweenx;
    EasyCallable const findobj;
    EasyCallable const flag;
    EasyCallable const gca;
    EasyCallable const gcf;
    EasyCallable const gci;
    EasyCallable const get;
    EasyCallable const get_backend;
    EasyCallable const get_cmap;
    EasyCallable const get_current_fig_manager;
    EasyCallable const get_figlabels;
    EasyCallable const get_fignums;
    EasyCallable const get_plot_commands;
    EasyCallable const get_scale_docs;
    EasyCallable const get_scale_names;
    EasyCallable const getp;
    EasyCallable const ginput;
    EasyCallable const gray;
    EasyCallable const grid;
    EasyCallable const hexbin;
    EasyCallable const hist;
    EasyCallable const hist2d;
    EasyCallable const hlines;
    EasyCallable const hold;
    EasyCallable const hot;
    EasyCallable const hsv;
    EasyCallable const imread;
    EasyCallable const imsave;
    EasyCallable const imshow;
    EasyCallable const interactive;
    EasyCallable const ioff;
    EasyCallable const ion;
    EasyCallable const is_numlike;
    EasyCallable const is_string_like;
    EasyCallable const ishold;
    EasyCallable const isinteractive;
    EasyCallable const jet;
    EasyCallable const legend;
    EasyCallable const locator_params;
    EasyCallable const loglog;
    EasyCallable const magnitude_spectrum;
    EasyCallable const margins;
    EasyCallable const matshow;
    EasyCallable const minorticks_off;
    EasyCallable const minorticks_on;
    EasyCallable const new_figure_manager;
    EasyCallable const normalize;
    EasyCallable const over;
    EasyCallable const pause;
    EasyCallable const pcolor;
    EasyCallable const pcolormesh;
    EasyCallable const phase_spectrum;
    EasyCallable const pie;
    EasyCallable const pink;
    EasyCallable const plot;
    EasyCallable const plot_date;
    EasyCallable const plotfile;
    EasyCallable const plotting;
    EasyCallable const polar;
    EasyCallable const prism;
    EasyCallable const psd;
    EasyCallable const pylab_setup;
    EasyCallable const quiver;
    EasyCallable const quiverkey;
    EasyCallable const rc;
    EasyCallable const rc_context;
    EasyCallable const rcdefaults;
    EasyCallable const register_cmap;
    EasyCallable const rgrids;
    EasyCallable const savefig;
    EasyCallable const sca;
    EasyCallable const scatter;
    EasyCallable const sci;
    EasyCallable const semilogx;
    EasyCallable const semilogy;
    EasyCallable const set_cmap;
    EasyCallable const setp;
    EasyCallable const show;
    EasyCallable const specgram;
    EasyCallable const spectral;
    EasyCallable const spring;
    EasyCallable const spy;
    EasyCallable const stackplot;
    EasyCallable const stem;
    EasyCallable const step;
    EasyCallable const streamplot;
    EasyCallable const subplot;
    EasyCallable const subplot2grid;
    EasyCallable const subplot_tool;
    EasyCallable const subplots;
    EasyCallable const subplots_adjust;
    EasyCallable const summer;
    EasyCallable const suptitle;
    EasyCallable const switch_backend;
    EasyCallable const table;
    EasyCallable const text;
    EasyCallable const thetagrids;
    EasyCallable const tick_params;
    EasyCallable const ticklabel_format;
    EasyCallable const tight_layout;
    EasyCallable const title;
    EasyCallable const tricontour;
    EasyCallable const tricontourf;
    EasyCallable const tripcolor;
    EasyCallable const triplot;
    EasyCallable const twinx;
    EasyCallable const twiny;
    EasyCallable const violinplot;
    EasyCallable const vlines;
    EasyCallable const waitforbuttonpress;
    EasyCallable const winter;
    EasyCallable const xcorr;
    EasyCallable const xkcd;
    EasyCallable const xlabel;
    EasyCallable const xlim;
    EasyCallable const xscale;
    EasyCallable const xticks;
    EasyCallable const ylabel;
    EasyCallable const ylim;
    EasyCallable const yscale;
    EasyCallable const yticks;

    Pyplot();

    auto render() -> cv::Mat;
    auto renderAndClose() -> cv::Mat;
    void saveAndClose(boost::filesystem::path const& path);
    void showAndClose();

    static
    void quickPlot(cv::InputArray src, std::string const& title="");

};

void plot(cv::InputArray src, std::string const& title="");
}

#endif /* PYTHON_PYPLOT_HPP_ */
