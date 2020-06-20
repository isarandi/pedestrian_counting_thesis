#include "Pyplot.hpp"

using namespace cv;
using namespace pyx;

void pyx::
plot(cv::InputArray src, std::string const& title)
{
    Pyplot plt;
    plt.plot(src);
    plt.title(title);
    plt.show();
    plt.close();
}

void Pyplot::
quickPlot(cv::InputArray src, std::string const& title)
{
    Pyplot plt;
    plt.plot(src);
    plt.title(title);
    plt.show();
    plt.close();
}

auto Pyplot::
render() -> cv::Mat
{
    auto p =
            boost::filesystem::temp_directory_path() /
            boost::filesystem::unique_path("%%%%%%%%%.png");
    savefig(p.string());
    auto img = cv::imread(p.string());
    boost::filesystem::remove(p);

    return img;
}

void Pyplot::
saveAndClose(boost::filesystem::path const& path)
{
    savefig(path.string());

    pyx::subst(
            "pickle.dump(_0, open(_1, 'wb'))",
            gcf(),
            path.string()+".pck");
    close();
}

void Pyplot::
showAndClose()
{
    show();
    close();
}

auto Pyplot::
renderAndClose() -> cv::Mat
{
    cv::Mat img = render();
    close();
    return img;
}

Pyplot::Pyplot()
: acorr("matplotlib.pylab", "acorr")
, angle_spectrum("matplotlib.pylab", "angle_spectrum")
, annotate("matplotlib.pylab", "annotate")
, arrow("matplotlib.pylab", "arrow")
, autoscale("matplotlib.pylab", "autoscale")
, autumn("matplotlib.pylab", "autumn")
, axes("matplotlib.pylab", "axes")
, axhline("matplotlib.pylab", "axhline")
, axhspan("matplotlib.pylab", "axhspan")
, axis("matplotlib.pylab", "axis")
, axvline("matplotlib.pylab", "axvline")
, axvspan("matplotlib.pylab", "axvspan")
, bar("matplotlib.pylab", "bar")
, barbs("matplotlib.pylab", "barbs")
, barh("matplotlib.pylab", "barh")
, bone("matplotlib.pylab", "bone")
, box("matplotlib.pylab", "box")
, boxplot("matplotlib.pylab", "boxplot")
, broken_barh("matplotlib.pylab", "broken_barh")
, cla("matplotlib.pylab", "cla")
, clabel("matplotlib.pylab", "clabel")
, clf("matplotlib.pylab", "clf")
, clim("matplotlib.pylab", "clim")
, close("matplotlib.pylab", "close")
, cohere("matplotlib.pylab", "cohere")
, colorbar("matplotlib.pylab", "colorbar")
, colormaps("matplotlib.pylab", "colormaps")
, colors("matplotlib.pylab", "colors")
, connect("matplotlib.pylab", "connect")
, contour("matplotlib.pylab", "contour")
, contourf("matplotlib.pylab", "contourf")
, cool("matplotlib.pylab", "cool")
, copper("matplotlib.pylab", "copper")
, csd("matplotlib.pylab", "csd")
, dedent("matplotlib.pylab", "dedent")
, delaxes("matplotlib.pylab", "delaxes")
, disconnect("matplotlib.pylab", "disconnect")
, draw("matplotlib.pylab", "draw")
, draw_if_interactive("matplotlib.pylab", "draw_if_interactive")
, errorbar("matplotlib.pylab", "errorbar")
, eventplot("matplotlib.pylab", "eventplot")
, figaspect("matplotlib.pylab", "figaspect")
, figimage("matplotlib.pylab", "figimage")
, figlegend("matplotlib.pylab", "figlegend")
, fignum_exists("matplotlib.pylab", "fignum_exists")
, figtext("matplotlib.pylab", "figtext")
, figure("matplotlib.pylab", "figure")
, fill("matplotlib.pylab", "fill")
, fill_between("matplotlib.pylab", "fill_between")
, fill_betweenx("matplotlib.pylab", "fill_betweenx")
, findobj("matplotlib.pylab", "findobj")
, flag("matplotlib.pylab", "flag")
, gca("matplotlib.pylab", "gca")
, gcf("matplotlib.pylab", "gcf")
, gci("matplotlib.pylab", "gci")
, get("matplotlib.pylab", "get")
, get_backend("matplotlib.pylab", "get_backend")
, get_cmap("matplotlib.pylab", "get_cmap")
, get_current_fig_manager("matplotlib.pylab", "get_current_fig_manager")
, get_figlabels("matplotlib.pylab", "get_figlabels")
, get_fignums("matplotlib.pylab", "get_fignums")
, get_plot_commands("matplotlib.pylab", "get_plot_commands")
, get_scale_docs("matplotlib.pylab", "get_scale_docs")
, get_scale_names("matplotlib.pylab", "get_scale_names")
, getp("matplotlib.pylab", "getp")
, ginput("matplotlib.pylab", "ginput")
, gray("matplotlib.pylab", "gray")
, grid("matplotlib.pylab", "grid")
, hexbin("matplotlib.pylab", "hexbin")
, hist("matplotlib.pylab", "hist")
, hist2d("matplotlib.pylab", "hist2d")
, hlines("matplotlib.pylab", "hlines")
, hold("matplotlib.pylab", "hold")
, hot("matplotlib.pylab", "hot")
, hsv("matplotlib.pylab", "hsv")
, imread("matplotlib.pylab", "imread")
, imsave("matplotlib.pylab", "imsave")
, imshow("matplotlib.pylab", "imshow")
, interactive("matplotlib.pylab", "interactive")
, ioff("matplotlib.pylab", "ioff")
, ion("matplotlib.pylab", "ion")
, is_numlike("matplotlib.pylab", "is_numlike")
, is_string_like("matplotlib.pylab", "is_string_like")
, ishold("matplotlib.pylab", "ishold")
, isinteractive("matplotlib.pylab", "isinteractive")
, jet("matplotlib.pylab", "jet")
, legend("matplotlib.pylab", "legend")
, locator_params("matplotlib.pylab", "locator_params")
, loglog("matplotlib.pylab", "loglog")
, magnitude_spectrum("matplotlib.pylab", "magnitude_spectrum")
, margins("matplotlib.pylab", "margins")
, matshow("matplotlib.pylab", "matshow")
, minorticks_off("matplotlib.pylab", "minorticks_off")
, minorticks_on("matplotlib.pylab", "minorticks_on")
, new_figure_manager("matplotlib.pylab", "new_figure_manager")
, normalize("matplotlib.pylab", "normalize")
, over("matplotlib.pylab", "over")
, pause("matplotlib.pylab", "pause")
, pcolor("matplotlib.pylab", "pcolor")
, pcolormesh("matplotlib.pylab", "pcolormesh")
, phase_spectrum("matplotlib.pylab", "phase_spectrum")
, pie("matplotlib.pylab", "pie")
, pink("matplotlib.pylab", "pink")
, plot("matplotlib.pylab", "plot")
, plot_date("matplotlib.pylab", "plot_date")
, plotfile("matplotlib.pylab", "plotfile")
, plotting("matplotlib.pylab", "plotting")
, polar("matplotlib.pylab", "polar")
, prism("matplotlib.pylab", "prism")
, psd("matplotlib.pylab", "psd")
, pylab_setup("matplotlib.pylab", "pylab_setup")
, quiver("matplotlib.pylab", "quiver")
, quiverkey("matplotlib.pylab", "quiverkey")
, rc("matplotlib.pylab", "rc")
, rc_context("matplotlib.pylab", "rc_context")
, rcdefaults("matplotlib.pylab", "rcdefaults")
, register_cmap("matplotlib.pylab", "register_cmap")
, rgrids("matplotlib.pylab", "rgrids")
, savefig("matplotlib.pylab", "savefig")
, sca("matplotlib.pylab", "sca")
, scatter("matplotlib.pylab", "scatter")
, sci("matplotlib.pylab", "sci")
, semilogx("matplotlib.pylab", "semilogx")
, semilogy("matplotlib.pylab", "semilogy")
, set_cmap("matplotlib.pylab", "set_cmap")
, setp("matplotlib.pylab", "setp")
, show("matplotlib.pylab", "show")
, specgram("matplotlib.pylab", "specgram")
, spectral("matplotlib.pylab", "spectral")
, spring("matplotlib.pylab", "spring")
, spy("matplotlib.pylab", "spy")
, stackplot("matplotlib.pylab", "stackplot")
, stem("matplotlib.pylab", "stem")
, step("matplotlib.pylab", "step")
, streamplot("matplotlib.pylab", "streamplot")
, subplot("matplotlib.pylab", "subplot")
, subplot2grid("matplotlib.pylab", "subplot2grid")
, subplot_tool("matplotlib.pylab", "subplot_tool")
, subplots("matplotlib.pylab", "subplots")
, subplots_adjust("matplotlib.pylab", "subplots_adjust")
, summer("matplotlib.pylab", "summer")
, suptitle("matplotlib.pylab", "suptitle")
, switch_backend("matplotlib.pylab", "switch_backend")
, table("matplotlib.pylab", "table")
, text("matplotlib.pylab", "text")
, thetagrids("matplotlib.pylab", "thetagrids")
, tick_params("matplotlib.pylab", "tick_params")
, ticklabel_format("matplotlib.pylab", "ticklabel_format")
, tight_layout("matplotlib.pylab", "tight_layout")
, title("matplotlib.pylab", "title")
, tricontour("matplotlib.pylab", "tricontour")
, tricontourf("matplotlib.pylab", "tricontourf")
, tripcolor("matplotlib.pylab", "tripcolor")
, triplot("matplotlib.pylab", "triplot")
, twinx("matplotlib.pylab", "twinx")
, twiny("matplotlib.pylab", "twiny")
, violinplot("matplotlib.pylab", "violinplot")
, vlines("matplotlib.pylab", "vlines")
, waitforbuttonpress("matplotlib.pylab", "waitforbuttonpress")
, winter("matplotlib.pylab", "winter")
, xcorr("matplotlib.pylab", "xcorr")
, xkcd("matplotlib.pylab", "xkcd")
, xlabel("matplotlib.pylab", "xlabel")
, xlim("matplotlib.pylab", "xlim")
, xscale("matplotlib.pylab", "xscale")
, xticks("matplotlib.pylab", "xticks")
, ylabel("matplotlib.pylab", "ylabel")
, ylim("matplotlib.pylab", "ylim")
, yscale("matplotlib.pylab", "yscale")
, yticks("matplotlib.pylab", "yticks")
{}
