import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
from matplotlib import cm
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
import pandas as pd
import warnings
from sklearn.preprocessing import MinMaxScaler
import pickle
import shared_config

def convert_ticks_in_day_to_timestamps(list_of_original_ticks, spacing_in_seconds):
    """

    :param list_of_original_ticks: e.g list(range(24))
    :param spacing_in_seconds: ['00:00',,,,,'00:05',,,,,]
    :return:
    """
    how_many_seconds = 3600 * 24
    seconds_per_original_tick = how_many_seconds // (len(list_of_original_ticks))
    list_of_ticks = []
    for i in range(0, how_many_seconds, seconds_per_original_tick):
        hours = i // 3600
        minutes = (i % 3600) // 60
        list_of_ticks.append("{:02d}".format(hours) + ":" + "{:02d}".format(minutes))

    prev = -1
    for i in range(len(list_of_ticks)):
        if (i * seconds_per_original_tick) // spacing_in_seconds <= prev:
            list_of_ticks[i] = ""
        else:
            prev = (i * seconds_per_original_tick) // spacing_in_seconds

    return list_of_ticks


def raw_data_to_one_d_signals_new_api(
    plotting,
    variable=None,
    DPI=300,
    input_file_name=None,
    output_path=None,
    station_dict=None,
    plotting_filter_type="mean",
    maximum_value_set_to_1=False,
    smoothing_window=30,
    smoothing_level=1,
    filter_station_ids=None,
    weekday_or_weekend = shared_config.weekday_or_weekend
):
    assert  weekday_or_weekend in ["weekday", "weekend"] or str(weekday_or_weekend) in [str(x) for x in list(range(7))]

    if ".pickle" in input_file_name or ".pkl" in input_file_name:
        # implies the pre-processed file obtained, then we can directly load from pickle
        with open(input_file_name, "rb") as handle:
            data = {}
            data_station_var_date = pickle.load(handle)
            for key in data_station_var_date:
                station, var, date_ = key
                if var != "Volume":
                    continue
                if station not in filter_station_ids:
                    continue
                day_of_week = pd.to_datetime(date_).dayofweek
                if (shared_config.single_day == -1 and ((weekday_or_weekend == "weekday" and day_of_week in [0, 1, 2, 3, 4]) or \
                            (weekday_or_weekend == "weekend" and day_of_week in [5, 6]) or \
                                str(weekday_or_weekend) == str(day_of_week))) \
                        or   (shared_config.single_day != -1 and    ( day_of_week == shared_config.single_day) ):

                    if station not in data:
                        data[station] = [data_station_var_date[key]]
                    else:
                        data[station].append(data_station_var_date[key])

            for key in data:
                # take the mean across days (weekday/weekend)
                array_ = np.array(data[key])
                data[key] = np.mean(array_, axis=0) # take mean across days
                debug_stop = True


            # maxval no longer being used; we just pass -1 here
            # this part was being used in the past to debug by plotting the values
            # w.r.t the maximum value
            return data, -1

    else:
        # implies csv file was obtained (old format)
        data = pd.read_csv(input_file_name)
    station_wise_data = {}
    plt.clf()

    if filter_station_ids is not None:
        filter_station_ids = set(filter_station_ids)

    for key in station_dict:
        if filter_station_ids is not None:
            if key not in filter_station_ids:
                continue

        for sensor_id in station_dict[key]["detectors"]:
            if key in station_wise_data:
                station_wise_data[key] += data[str(sensor_id)]
            else:
                station_wise_data[key] = data[str(sensor_id)]

            if plotting:
                this_sensor = str(sensor_id)

                try:
                    plt.plot(
                        smoothing(
                            data[this_sensor],
                            smoothing_level=smoothing_level,
                            smoothing_window=smoothing_window,
                            plotting_filter_type=plotting_filter_type,
                        ),
                        label=this_sensor,
                    )

                except:
                    print("Something wrong with sensor: ", this_sensor)

    if plotting:
        # plt show outside the if loop to put all sensors on the same map
        plt.xlabel("Time of Day")
        plt.xticks(
            range(1440 * 2),
            convert_ticks_in_day_to_timestamps(range(1440 * 2), spacing_in_seconds=120 * 60),
            rotation=45,
        )
        plt.ylabel(variable)
        plt.title(
            "Flows from various nearby sensors in Minnesota\n moving average "
            + str(smoothing_window)
            + " minutes\n Multiple rounds of smoothing"
            + str(smoothing_level),
            fontsize=7,
        )
        plt.legend(fontsize=4, bbox_to_anchor=(1.1, 1))
        plt.savefig(output_path + "smoothing_filter_" + plotting_filter_type + "_" + variable + ".png", dpi=DPI)
        plt.show(block=False)

    maxVal = -1
    for station_id in station_wise_data:
        if np.max(station_wise_data[station_id]) > maxVal:
            maxVal = np.max(station_wise_data[station_id])

    if maximum_value_set_to_1:
        for station_id in station_wise_data:
            station_wise_data[station_id] /= maxVal

    if plotting:
        #  plot stations
        def plot_volume_station_wise(smoothing_window_inner):
            counter = 1
            for station_id in station_wise_data:
                plt.plot(
                    smoothing(
                        station_wise_data[station_id],
                        smoothing_level=1,
                        smoothing_window=smoothing_window_inner,
                        plotting_filter_type="mean",
                    ),
                    label=station_id,
                    alpha=0.2 * counter,
                )
                counter += 1

            plt.xlabel("Time of Day (minutes)")
            plt.ylabel(variable)

            # use actual time on x-axis
            plt.xticks(
                range(1440 * 2),
                convert_ticks_in_day_to_timestamps(range(1440 * 2), spacing_in_seconds=120 * 60),
                rotation=45,
            )

            plt.title(
                "Flows from various stations in Minnesota\n moving average "
                + str(smoothing_window_inner)
                + " minutes\n Multiple rounds of smoothing"
                + str(smoothing_level),
                fontsize=7,
            )
            plt.legend(fontsize=4, bbox_to_anchor=(1.1, 1))
            plt.savefig(
                output_path
                + "_"
                + plotting_filter_type
                + "_"
                + "_stations_"
                + variable
                + "_smoothing_window_"
                + str(smoothing_window_inner)
                + ".png",
                dpi=DPI,
            )
            plt.show(block=False)

        plot_volume_station_wise(1)
        plot_volume_station_wise(10)
        plot_volume_station_wise(30)

    return station_wise_data, maxVal


"""
Copied from https://gist.github.com/bhawkins/3535131
"""


def medfilt(x, k):
    """Apply a length-k median filter to a 1D array x.
    Boundaries are extended by repeating endpoints.
    """
    assert k % 2 == 1, "Median filter length must be odd."
    assert x.ndim == 1, "Input must be one-dimensional."
    k2 = (k - 1) // 2
    y = np.zeros((len(x), k), dtype=x.dtype)
    y[:, k2] = x
    for i in range(k2):
        j = k2 - i
        y[j:, i] = x[:-j]
        y[:j, i] = x[0]
        y[:-j, -(i + 1)] = x[j:]
        y[-j:, -(i + 1)] = x[-1]
    return np.median(y, axis=1)


def smoothing(x, smoothing_level, smoothing_window, plotting_filter_type="mean"):
    if smoothing_window == 1 and smoothing_level == 1:
        # no change
        return x

    for _ in range(smoothing_level):
        if plotting_filter_type == "mean":
            a = np.convolve(x, np.ones(smoothing_window) / smoothing_window)
        elif plotting_filter_type == "median":
            a = medfilt(x, smoothing_window // 2 * 2 + 1)
    return a


if __name__ == "__main__":
    input = list(range(48))
    ticks = convert_ticks_in_day_to_timestamps(input, spacing_in_seconds=10)
    print(ticks)
    print(input)
    assert len(ticks) == len(input)
