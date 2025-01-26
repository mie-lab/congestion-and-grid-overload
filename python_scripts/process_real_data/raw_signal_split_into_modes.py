import os.path

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
from matplotlib import cm
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
import pandas as pd
import warnings
from sklearn.preprocessing import MinMaxScaler

import shared_config


def raw_data_to_one_d_signals_for_congestion_grid_overload():
    """
    Data format in csv:

    sensors	variable	date	12:00:30 AM	12:01 AM	12:01:30 AM	12:02 AM
    130	Flow	01.08.21	0	0	0	0
    130	Flow	02.08.21	00.01.00	00.01.00	00.01.00	00.01.00
    130	Flow	03.08.21	0	0	0	0
    130	Flow	04.08.21	0	0	0	0
    130	Flow	05.08.21	0	0	0	0
    130	Flow	06.08.21	0	0	0	0
    130	Flow	07.08.21	0	0	0	0
    141	Flow	01.08.21	0	0	0	0
    :return:
    """
    data, max_val = raw_data_to_one_d_signals_minnst(
        dates=["01.08.21"],
        transpose_needed=False,
        plotting=True,
        variable="Flow",
        two_rounds_of_scaling=0,
        resolution=500,
        output_path=os.path.join(shared_config.BASE_FOLDER_with_repo_name, "output_images/raw_data/Sunday_August_1-"),
    )


def raw_data_to_one_d_signals_minnst(
    dates,
    plotting,
    variable,
    two_rounds_of_scaling,
    DPI=300,
    input_file_name=None,
    output_path=None,
    transpose_needed=True,
    resolution=300,
    sensors="ALL",
):
    """

    :param resolution: in seconds
    :param sensors: list of sensors (integer)
    :param dates: list of dates (integer)
    :param input_file_name: string filename relative to data
    :param plotting: T/F
    :param output_path:
    :param variable:
    :param DPI:
    :param two_rounds_of_scaling: 0: False, 1. max for traffic variable;  2. sklearn.minMaxScaler for WS/SS
    :return:
    """

    """
    raw data looks as follows: (first three rows are sensor, variable, date) 
    ['', ' ', '1668', '1668']
    ['', ' ', 'Capacity', 'Capacity']
    ['', ' ', '2021/08/01', '2021/08/02']
    ['', '12:00:30 AM', '1800.0', '1800.0']
    ['', '12:01 AM', '1680.0', '1800.0']
    ['', '12:01:30 AM', '1800.0', '1680.0']
    """

    # After transposing, it looks as follows:
    """
    # @first row empty		# @second row time labels		
                12:00:30 AM	12:01 AM	12:01:30 AM
    1668	Capacity	01.08.21	1800	1680	1800
    1668	Capacity	02.08.21	1800	1800	1680
    1668	Capacity	03.08.21	1800	1800	1800
    1668	Capacity	04.08.21	1800	1800	1800
    1668	Capacity	05.08.21	1800	1800	1800
    1668	Capacity	06.08.21	1800	1800	1800
    1668	Capacity	07.08.21	1800	1800	1800
    
    """

    data = {}

    if transpose_needed:
        pd.read_csv("data/" + input_file_name + resolution + ".csv", header=None).T.to_csv(
            "data/transposed_input.csv", header=False, index=False
        )

    counter = 0
    with open("data/transposed_input.csv") as f:
        next(f)  # @first row empty
        # next(f)  # @second row time labels

        for row in f:
            listed = row.strip().split(",")
            this_sensor, this_variable, this_date = listed[0:3]
            counter += 1
            try:
                values = np.array(list(map(float, listed[3:])))
            except:
                warnings.warn("Error in line" + str(counter))
                continue

            data[this_sensor, this_variable, this_date] = values

    set_of_dates = set(dates)

    extracted_1_d_signals = {}
    max_val_variable = -999999

    unique_sensors = []
    for (this_sensor, this_variable, this_date) in data:
        unique_sensors.append(this_sensor)
        if this_date in set_of_dates and this_variable == variable:
            if max(data[this_sensor, this_variable, this_date]) > max_val_variable:
                max_val_variable = max(data[this_sensor, this_variable, this_date])

            if two_rounds_of_scaling == 0:
                extracted_1_d_signals[this_sensor] = data[this_sensor, this_variable, this_date]
            elif two_rounds_of_scaling == 1:
                # Step 1:
                step_1_scaled_data = np.array(data[this_sensor, this_variable, this_date]) / max_val_variable
                extracted_1_d_signals[this_sensor] = step_1_scaled_data
            elif two_rounds_of_scaling == 2:

                # Step 1
                step_1_scaled_data = np.array(data[this_sensor, this_variable, this_date]) / max_val_variable

                # Step 2:
                scaler = MinMaxScaler()
                extracted_1_d_signals[this_sensor] = scaler.fit_transform(step_1_scaled_data.reshape(-1, 1))
            else:
                print("\n\nERROR in scaling levels; Mustbe 0 or 1 or 2\n\n")
                return -1

    unique_sensors = list(set(unique_sensors))
    # Allow different series of colors based on types of sensors (S vs L)
    S_list = []
    L_list = []
    for key in unique_sensors:
        if "S" in key:
            S_list.append(key)
        else:
            L_list.append(key)
    list.sort(S_list)
    list.sort(L_list)
    unique_sensors = S_list + L_list

    color_func_S_list = cm.get_cmap("Blues", 10)
    color_func_L_list = cm.get_cmap("Oranges", 20)

    c_map = {}
    for i in range(len(S_list)):
        c_map[S_list[i]] = color_func_S_list(1 / len(S_list) * (i + 1))
    for i in range(len(L_list)):
        c_map[L_list[i]] = color_func_L_list(1 / len(L_list) * (i + 1))

    smoothing_window = 30
    smoothing_level = 1
    if plotting:
        plt.clf()
        if sensors == "ALL":
            sensors = unique_sensors

        for this_sensor in sensors:
            this_sensor = str(this_sensor)

            try:
                plt.plot(
                    convert_30_seconds_to_minute_wise(
                        extracted_1_d_signals[this_sensor],
                        smoothing_level=smoothing_level,
                        smoothing_window=smoothing_window,
                    ),
                    label=this_sensor,
                    color=c_map[this_sensor],
                )
            except:
                print("Something wrong with sensor: ", this_sensor)

        plt.xlabel("Time of Day (minutes)")
        plt.ylabel("Hourly flow")
        plt.title(
            "Flows from various nearby sensors in Minnesota\n moving average "
            + str(smoothing_window)
            + " minutes\n Multiple rounds of smoothing"
            + str(smoothing_level),
            fontsize=7,
        )
        plt.legend(fontsize=4, bbox_to_anchor=(1.1, 1))
        plt.savefig(output_path + variable + ".png", dpi=resolution)
        plt.show(block=False)

    return extracted_1_d_signals, max_val_variable


def convert_30_seconds_to_minute_wise(x, smoothing_level, smoothing_window):
    a = []

    # 30 secs to 1 min
    for i in range(0, len(x), 2):
        a.append(x[i] + x[i + 1])

    for _ in range(smoothing_level):
        a = np.convolve(a, np.ones(smoothing_window) / smoothing_window)

    return a


def test_S_and_L_add_up(output_path):  # test for issue #84
    """

    :param output_path:
    :return:
    """
    for day, date in [
        ["Sunday", "01.08.21"],
        ["Monday", "02.08.21"],
        ["Tuesday", "03.08.21"],
        ["Wednesday", "04.08.21"],
        ["Thursday", "05.08.21"],
        ["Sunday", "06.08.21"],
        ["Saturday", "07.08.21"],
    ]:
        data, max_val = raw_data_to_one_d_signals_minnst(
            dates=[date],
            transpose_needed=False,
            plotting=False,
            variable="Flow",
            two_rounds_of_scaling=0,
            resolution=500,
            output_path=os.path.join(shared_config.BASE_FOLDER_with_repo_name, "output_images/raw_data/")
            + day
            + "-",
        )

        # Is S109 = 413+414+415+419?
        S109 = data["S109"]
        L413 = data["413"]
        L414 = data["414"]
        L415 = data["415"]
        L419 = data["419"]
        lanes_sum = np.sum(L413 + L414 + L415 + L419)
        print(day, date)
        print(
            "Total difference: L ",
            lanes_sum,
            "vs",
            np.sum(S109),
            " % diff=",
            round(abs(lanes_sum - np.sum(S109)) / lanes_sum * 100, 2),
        )
        plt.clf()

        plt.plot(S109, label="S109", alpha=0.9, linewidth=0.9, color="red")
        plt.plot(L413 + L414 + L415 + L419, label="L413 + L414 + L415 + L419", color="blue", alpha=0.2, linewidth=0.3)
        plt.plot(S109 - (L413 + L414 + L415 + L419), label="S109-(L413+L414+L415+L419)", linewidth=0.3, color="green")
        plt.title("Absolute differences for S109")
        plt.legend(fontsize=5)
        plt.savefig(output_path + "S109_flow_difference" + day + date + ".png", dpi=300)
        plt.show(block=False)

        # Is S1817 = 7241+7242+7243+7244


if __name__ == "__main__":
    for day, date in [
        ["Sunday", "01.08.21"],
        ["Monday", "02.08.21"],
        ["Tuesday", "03.08.21"],
        ["Wednesday", "04.08.21"],
        ["Thursday", "05.08.21"],
        ["Sunday", "06.08.21"],
        ["Saturday", "07.08.21"],
    ]:
        data, max_val = raw_data_to_one_d_signals_minnst(
            dates=[date],
            transpose_needed=False,
            plotting=True,
            variable="Volume",
            two_rounds_of_scaling=0,
            resolution=500,
            output_path=os.path.join(shared_config.BASE_FOLDER_with_repo_name, "output_images/raw_data/")
            + day
            + "-",
        )

    print(day, date)
    test_S_and_L_add_up(
        output_path=os.path.join(shared_config.BASE_FOLDER_with_repo_name, "output_images/raw_data/")
    )
