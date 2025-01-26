import matplotlib.pyplot as plt
import os

import numpy as np


os.chdir("./")
output_path = "output_images"


def histogram_of_occupancy_and_volume(data, station_id, output_image_path, title, plotting):
    if not plotting:
        # just a quick workaround to skip this function when we are not using it
        return
    x = (data[station_id]).to_numpy().flatten()
    plt.hist(x, 20)
    plt.title(title)
    plt.xlabel("Bins")
    plt.ylabel("Bin counts")
    plt.savefig(output_image_path + "/" + title + ".png")
    plt.show(block=False)


def plot_volume_and_occupancy_together(data_vol, data_occupancy, station_id, output_image_path, date_, ylim, prefix):
    """

    :param data_vol:
    :param data_occupancy:
    :param station_id:
    :param output_image_path:
    :param date_:
    :param ylim:
    :param prefix:
    :return:
    """
    v = (data_vol[station_id]).to_numpy().flatten()
    o = (data_occupancy[station_id]).to_numpy().flatten()
    for smoothing_window in [1, 5, 10, 30, 50]:
        plt.plot(np.convolve(v, (1 / smoothing_window) * np.ones(smoothing_window)), label="Volume")
        plt.plot(np.convolve(o, (1 / smoothing_window) * np.ones(smoothing_window)), label="Occupancy")
        plt.xticks(
            range(1440 * 2),
            convert_ticks_in_day_to_timestamps(range(1440 * 2), spacing_in_seconds=120 * 60),
            rotation=45,
        )
        plt.ylim(0, ylim)
        plt.title(prefix + "_at_station" + station_id)
        plt.legend()
        plt.savefig(
            output_image_path
            + "/"
            + prefix
            + "_at_station"
            + station_id
            + "_"
            + date_
            + "_smoothing_"
            + str(smoothing_window)
            + ".png"
        )
        plt.xlabel("Time of day")
        plt.show(block=False)


def plot_volume_vs_occupancy_together(
    data_vol,
    data_occupancy,
    station_id,
    output_image_path,
    date_,
    ylim,
    prefix,
    invert,
    plot_both_together_to_check_turbulence,
):
    """

    :param data_vol:
    :param data_occupancy:
    :param station_id:
    :param output_image_path:
    :param date_:
    :param ylim:
    :param prefix:
    :param invert:
    :param plot_both_together_to_check_turbulence:
    :return:
    """

    v = (data_vol[station_id]).to_numpy().flatten()
    o = (data_occupancy[station_id]).to_numpy().flatten()
    for smoothing_window in [1, 5, 10, 30, 50]:
        if not invert:
            plt.scatter(
                np.convolve(v, (1 / smoothing_window) * np.ones(smoothing_window)),
                np.convolve(o, (1 / smoothing_window) * np.ones(smoothing_window)),
                s=0.2,
            )
            ylabel, xlabel = "Volume", "occupancy"

        if invert:
            plt.scatter(
                np.convolve(o, (1 / smoothing_window) * np.ones(smoothing_window)),
                np.convolve(v, (1 / smoothing_window) * np.ones(smoothing_window)),
                s=0.2,
            )
            ylabel, xlabel = "occupancy", "Volume"

        if plot_both_together_to_check_turbulence:
            if not invert:
                plt.scatter(
                    np.convolve(v, (1 / smoothing_window) * np.ones(smoothing_window)),
                    np.convolve(o, (1 / smoothing_window) * np.ones(smoothing_window)),
                    s=0.2,
                )
                ylabel, xlabel = "Volume", "occupancy"

            if invert:
                plt.scatter(
                    np.convolve(o, (1 / smoothing_window) * np.ones(smoothing_window)),
                    np.convolve(v, (1 / smoothing_window) * np.ones(smoothing_window)),
                    s=0.2,
                )
                ylabel, xlabel = "occupancy", "Volume"
            # legend is needed in case we keep both plots together
            plt.legend(fontsize=5)

        if ylim != None:
            plt.ylim(0, ylim)
        plt.title(prefix + "_at_station" + station_id)
        plt.savefig(
            output_image_path
            + "/"
            + prefix
            + "_at_station"
            + station_id
            + "_"
            + date_
            + "_smoothing_"
            + str(smoothing_window)
            + ".png"
        )
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.gca().set_aspect("equal")
        plt.show(block=False)


if __name__ == "__main__":
    os.system("find ../output_images/ ! -type d -exec rm '{}' \;")

    volume_data = {}
    occupancy_data = {}
    for filename in [
        "30sec_volumes-20140801",
        "30sec_volumes-20140802",
        "30sec_volumes-20140803",
        "30sec_volumes-20140804",
        "30sec_volumes-20140805",
        "30sec_volumes-20140806",
        "30sec_volumes-20140807",
    ]:
        date_ = filename.split("-")[1]
        volume_data[date_], max_val_variable = raw_data_to_one_d_signals_new_api(
            plotting=False,
            plotting_filter_type="mean",
            DPI=500,
            input_file_name="../data/mnnsta/" + filename + ".csv",
            variable="Volume",
            station_dict={
                "S57": [315, 316, 317, 318, 6791],
                "S59": [323, 324, 325, 326, 6793],
                "S58": [319, 320, 321, 322, 6792],
            },
            output_path="../output_images/EDA_plots/raw_data/",
            maximum_value_set_to_1=False,
        )

        print("Volume max_val_variable: ", max_val_variable, "\n\n")

        for station_id in ["S57", "S58", "S59"]:
            histogram_of_occupancy_and_volume(
                volume_data[date_],
                station_id=station_id,
                output_image_path="../output_images/EDA_plots",
                title="Histogram for Volume at station: " + station_id + "_" + filename,
                plotting=False,
            )

    for filename in [
        "30sec_occ-20140801",
        "30sec_occ-20140802",
        "30sec_occ-20140803",
        "30sec_occ-20140804",
        "30sec_occ-20140805",
        "30sec_occ-20140806",
        "30sec_occ-20140807",
    ]:
        date_ = filename.split("-")[1]
        occupancy_data[date_], max_val_variable = raw_data_to_one_d_signals_new_api(
            plotting=False,
            plotting_filter_type="mean",
            DPI=500,
            input_file_name="../data/mnnsta/" + filename + ".csv",
            variable="Occupancy",
            station_dict={
                "S57": [315, 316, 317, 318, 6791],
                "S59": [323, 324, 325, 326, 6793],
                "S58": [319, 320, 321, 322, 6792],
            },
            output_path="../output_images/EDA_plots/raw_data",
            maximum_value_set_to_1=False,
        )
        print("Occupancy max_val_variable: ", max_val_variable, "\n\n")

        for station_id in ["S57", "S58", "S59"]:
            histogram_of_occupancy_and_volume(
                occupancy_data[date_],
                station_id=station_id,
                output_image_path="../output_images/EDA_plots",
                title="Histogram for Occupancy at station: " + station_id + "_" + filename,
                plotting=False,
            )

    for date_ in ["20140801", "20140802", "20140803", "20140804", "20140805", "20140806", "20140807"]:

        # plot flow and occupancy vs time together
        for station_id in ["S57", "S58", "S59"]:
            plot_volume_and_occupancy_together(
                data_vol=volume_data[date_],
                data_occupancy=occupancy_data[date_],
                station_id=station_id,
                output_image_path="../output_images/EDA_plots",
                date_=date_,
                ylim=600,
                prefix="Volume_and_Occupancy_at_station",
            )

        # plot flow vs occupancy
        for station_id in ["S57", "S58", "S59"]:
            plot_volume_vs_occupancy_together(
                data_vol=volume_data[date_],
                data_occupancy=occupancy_data[date_],
                station_id=station_id,
                output_image_path="../output_images/EDA_plots",
                date_=date_,
                ylim=None,
                prefix="Volume_vs_Occupancy_at_station",
                invert=False,
                plot_both_together_to_check_turbulence=True,
            )

        # plot occupancy vs flow
        for station_id in ["S57", "S58", "S59"]:
            plot_volume_vs_occupancy_together(
                data_vol=volume_data[date_],
                data_occupancy=occupancy_data[date_],
                station_id=station_id,
                output_image_path="../output_images/EDA_plots",
                date_=date_,
                ylim=None,
                prefix="Occupancy_vs_Volume_at_station",
                invert=True,
                plot_both_together_to_check_turbulence=True,
            )
