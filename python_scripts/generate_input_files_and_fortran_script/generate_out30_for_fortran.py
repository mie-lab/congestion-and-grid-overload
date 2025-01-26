import sys
import csv
import warnings

import matplotlib.pyplot as plt
import numpy as np

from python_scripts.process_real_data.raw_data_new_format.parse_xml_file import parse_xml_file
from python_scripts.process_real_data.raw_data_new_format.raw_signal_split_into_modes import (
    raw_data_to_one_d_signals_new_api,
)
from python_scripts.process_real_data.raw_signal_split_into_modes import raw_data_to_one_d_signals_minnst
import shared_config
from tqdm import tqdm

sys.path.append(shared_config.BASE_FOLDER_with_repo_name)


def generate_out_30_file(
    list_of_origins,
    timesteps,
    loading_type,
    y_lim_demand,
    params,
    seconds_offset_midnight,
    smoothing_data_window=None,
    uniform_demand_value=0.1,
):
    """

    Args:
      list_of_origins: as the name suggests
      timesteps: how many time steps does the loading happen (this was 3600 in Yi's original version)
      loading_type: as the name suggests
      demand_params = [gaussian_demand_max_val etc..]
      y_lim_demand: for plotting

    Returns:
      None

    """
    how_many_origins = len(list_of_origins)

    path_to_out_30_file = "input_output_text_files/out30.txt"

    with open(path_to_out_30_file, "w") as f:
        csvwriter = csv.writer(f, delimiter="	")

        #  multiplying by (1+np.random.rand()*0.1) introduces 10% stochasticity

        if loading_type == "uniform":
            total = 0
            for i in range(timesteps):
                demand_all_origins_each_time_step = []

                for j in range(how_many_origins):
                    demand_all_origins_each_time_step.append(uniform_demand_value * (1 + np.random.rand() * 0.01))
                total += sum(demand_all_origins_each_time_step)
                csvwriter.writerow(demand_all_origins_each_time_step)

            with open("output_images/total_arrival", "w") as f:
                csvwriter_record_total_demand = csv.writer(f)
                csvwriter_record_total_demand.writerow([total])

        elif loading_type == "binary":

            # high demand for half the day
            for i in range(timesteps // 2):
                demand_all_origins_each_time_step = []
                for j in range(how_many_origins):
                    demand_all_origins_each_time_step.append(1 * (1 + np.random.rand() * 0.05))
                csvwriter.writerow(demand_all_origins_each_time_step)

            # low demand for the other half the day
            for i in range(timesteps // 2, timesteps):
                demand_all_origins_each_time_step = []
                for j in range(how_many_origins):
                    demand_all_origins_each_time_step.append(0.01 * (1 + np.random.rand() * 0.05))
                csvwriter.writerow(demand_all_origins_each_time_step)

        elif loading_type == "gaussian":
            mu = 10.0
            sigma = 2.0

            data = np.random.randn(100000) * sigma + mu
            data = np.clip(data, 0, mu * 10)  # clip negative values

            hx, _, _ = plt.hist(data, bins=timesteps)

            # ensure max inflow from origin to be params[0]
            hx = hx / (np.max(hx)) * params[0]

            # add some uniform random to the gaussian
            hx = hx + np.random.rand(timesteps) * 0.01

            # overwrite the first few cells with zero (to remove the wrongly shown peak due to small values (read: outliers))
            hx[0:10] = 0

            total = 0
            for i in range(timesteps):
                demand_all_origins_each_time_step = []
                for j in range(how_many_origins):
                    demand_all_origins_each_time_step.append(hx[i] * (1 + np.random.rand() * 0.01))
                total += sum(demand_all_origins_each_time_step)
                csvwriter.writerow(demand_all_origins_each_time_step)

            with open("output_images/total_arrival", "w") as f:
                csvwriter_record_total_demand = csv.writer(f)
                csvwriter_record_total_demand.writerow([total])

        elif loading_type == "two_peaks":

            BINS = np.arange(0, 100 + 100 / timesteps, 100 / timesteps)

            mu = 20
            sigma = 2
            data = np.random.randn(100000) * sigma + mu
            data = np.clip(data, 0, 100)  # clip negative values
            hx, _, _ = plt.hist(data, bins=BINS)

            # ensure max inflow from origin to be params[0]
            hx = hx / (np.max(hx)) * params[0]

            # add some uniform random to the gaussian
            hx_1 = hx + np.random.rand(hx.shape[0]) * 0.01

            mu = 50
            sigma = 2
            data = np.random.randn(100000) * sigma + mu
            data = np.clip(data, 0, 100)  # clip negative values
            hx, _, _ = plt.hist(data, bins=BINS)
            hx = hx / (np.max(hx)) * params[0]
            hx_2 = hx + np.random.rand(hx.shape[0]) * 0.01

            # overwrite the first few cells with zero (to remove the wrongly shown peak due to small values (read: outliers))
            hx_2[0:10] = 0

            hx = hx_1 + hx_2

            total = 0
            for i in range(timesteps):
                demand_all_origins_each_time_step = []
                for j in range(how_many_origins):
                    demand_all_origins_each_time_step.append(hx[i] * (1 + np.random.rand() * 0.01))
                total += sum(demand_all_origins_each_time_step)
                csvwriter.writerow(demand_all_origins_each_time_step)

            with open("output_images/total_arrival", "w") as f:
                csvwriter_record_total_demand = csv.writer(f)
                csvwriter_record_total_demand.writerow([total])

        elif loading_type == "real_data":
            # day, date, OR_sensor_id_map = params[0], params[1], params[2]

            # ["Sunday", "01.08.21"],
            # ["Monday", "02.08.21"],{32:"S450", 233:"L3434",344:-999999}

            # data, max_val = raw_data_to_one_d_signals_minnst(
            #     dates=[date],
            #     transpose_needed=False,
            #     plotting=False,
            #     variable="Flow",
            #     two_rounds_of_scaling=0,
            # )

            #  format:
            """
            real_params=[
                            "data/mnnsta_data_new_api/30sec_volumes-20140801.csv",
                            {
                                1: {"S57", 1},
                                26: {"S57", 1},
                                62: {"S57", 1},
            """

            datacsvfile, OR_sensor_id_map = params[0], params[1]

            filter_station_ids = []
            for key in OR_sensor_id_map:
                filter_station_ids.append(OR_sensor_id_map[key][0])
            filter_station_ids = list(set(list(filter_station_ids)))

            # get data from new data api
            data, max_val = raw_data_to_one_d_signals_new_api(
                plotting=False,
                variable="Volume",
                DPI=300,
                input_file_name=datacsvfile,
                output_path=None,
                station_dict=parse_xml_file("python_scripts/process_real_data/raw_data_new_format/xml_file.xml"),
                plotting_filter_type="mean",
                maximum_value_set_to_1=False,
                smoothing_window=30,
                smoothing_level=1,
                filter_station_ids=filter_station_ids,
            )

            if smoothing_data_window:
                for sensor in data:
                    data = np.convolve(data[sensor], np.ones(int(smoothing_data_window * 60)))

            import matplotlib.pyplot as plt
            plt.clf()
            for key in data:
                plt.title("Raw data")
                plt.plot(data[key], label=key)
            plt.legend()
            plt.show()

            counter = 0
            for i in tqdm(range(timesteps), desc = "Generating input file (per second flow)"):
                demand_all_origins_each_time_step = []
                for OR in list_of_origins:
                    try:
                        sensor_id = OR_sensor_id_map[OR][0]
                        debug_fraction = float(OR_sensor_id_map[OR][1])
                    except KeyError:
                        # fill up all the remaining origins for which we don't have data
                        # To-do these zeros should be replaced with a fraction based on the
                        # number of ratio between merging and main highway
                        # however, it does not affect our results
                        counter += 1
                        sensor_id = -999999


                    if sensor_id == -999999:  # sensor not present case
                        demand_all_origins_each_time_step.append(0)
                    else:
                        x = data[sensor_id][(i + seconds_offset_midnight) // shared_config.raw_data_gran_seconds] * \
                            debug_fraction / shared_config.raw_data_gran_seconds
                        demand_all_origins_each_time_step.append(x)

                csvwriter.writerow(demand_all_origins_each_time_step)
            with open("input_output_text_files/count_of_missing_dicts.txt","w") as f2:
                csvwriter = csv.writer(f2)
                csvwriter.writerow(["Missing OR maps count = " + str(counter//timesteps) +" out of "+str(len(list_of_origins))])

        else:
            print("Wrong loading type provided \n\n\n")
            return -1

    max_y_lim = 0
    for i in range(len(list_of_origins)):
        with open(path_to_out_30_file) as f:

            demand_for_current_origin = []
            for row in f:
                listed = row.strip().split("	")
                try:
                    assert len(listed) == len(list_of_origins)
                except:
                    raise Exception(
                        "assert len(listed) == len(list_of_origins) failed: "
                        + str(len(listed))
                        + " vs "
                        + str(len(list_of_origins))
                    )
                    sys.exit(0)

                demand_for_current_origin.append(float(listed[i]))
            max_y_lim = max(max_y_lim, max(demand_for_current_origin))

            plt.plot(
                range(timesteps),
                demand_for_current_origin,
                label="Origin cell: " + str(list_of_origins[i]),
                linewidth=0.5,
            )

    plt.legend()
    plt.ylim(0, max_y_lim * 1.4)
    plt.title("Input demand (Out30 file) visualised")
    plt.ylabel("Outflow from origin cells")
    plt.xlabel(r"Time stamp: unit ($\Delta t$)")
    plt.ylim(0, y_lim_demand)
    plt.tight_layout()
    plt.grid(True)
    plt.savefig("output_images/demand_loading.png", dpi=300)
    plt.show(block=False)
    plt.close()

    return 0


if __name__ == "__main__":
    generate_out_30_file(10)
