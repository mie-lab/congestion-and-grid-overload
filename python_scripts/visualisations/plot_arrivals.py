import csv
import sys
import re
import networkx as nx
import numpy as np
from joblib import Parallel, delayed
import cv2
import glob
import imageio
import matplotlib
from datetime import datetime
import datetime as dt

# matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib import cm as colorMap
import os
from tqdm import tqdm

import shared_config
# from datetime import datetime as dt

sys.path.append(shared_config.BASE_FOLDER_with_repo_name)


def create_combined_destination_cells(outflow_dict, destination_merge_file_path):
    """

    :param outflow_dict:
    :param destination_merge_file_path:
    :return:
    """
    with open(destination_merge_file_path) as f:
        for row in f:

            # if the file is empty, we return the input dictionary as it is
            if len(row.strip()) == 0:
                return outflow_dict

            listed = row.strip().split("+")
            val = []
            for key in listed:
                val.append(outflow_dict[int(key)])
            val = np.array(val)
            outflow_dict[row] = np.sum(val, axis=0)
    return outflow_dict


"""
******** OUTFLOW START *******
Time Step     CELL17     CELL43     CELL70     CELL79     CELL88     CELL92     CELL92
--------------------------------------
1       0.12       1.27       0.00       0.00       0.00       0.00       0.00
2       0.00       0.91       0.00       0.00       0.00       0.00       0.00
"""


def process_outfl_file(plotting_enabled, cumulative_window, DPI, y_limit, grid=False):
    """

    Args:
        plotting_enabled: as the name suggests, plotting can be really slow when the flows are being extracted
        for a large number of cells

        cumulative_window: as the name suggests (unit: minutes)

        DPI, y_limit: as the name suggests

        Currently this reads from the hard coded path; TO-DO: need to change it to parameter

    Returns:
      outflows_cell_wise: per cell, per time step flow; cell_name is the key

    """

    flag_read_outflow = False  # initially, we set this to False

    time_stamps = []
    outflows_cell_wise = {}

    with open("input_output_text_files/outfl.txt") as f:
        num_lines = int(os.popen("wc -l input_output_text_files/outfl.txt").read().strip().split(" ")[0])

        for row in tqdm(f, desc="Processing outfl file... ", total=num_lines):

            row_strip = row.strip()
            if "OUTFLOW START" in row_strip:
                flag_read_outflow = True

                for row_skip in f:
                    # using regex to split instead of standard split so that we can split based on any delimiter
                    # not necessarilty four spaces, tab etc..

                    # Also, we use 2:end because the first two are 'Time' and 'Step'
                    list_of_output_cells = [
                        int(x.replace("CELL", "")) for x in (list(filter(None, re.split("[, \-!?:]+", row_skip))))[2:]
                    ]
                    for key in list_of_output_cells:
                        outflows_cell_wise[key] = []
                    print(list_of_output_cells)
                    break
                next(f)
                continue

            if "OUTFLOW END" in row:
                flag_read_outflow = False

            if flag_read_outflow == True:
                # using regex to split instead of standard split so that we can split based on any delimiter
                # not necessarilty four spaces, tab etc..

                listed = list(filter(None, re.split("[, \-!?:]+", row_strip)))
                time_stamps.append(int(listed[0]))
                for counter, key in enumerate(list_of_output_cells):

                    # +1 because the first column of every row is the time stamp
                    if listed[counter + 1] == "******":
                        listed[counter + 1] = 10 ** 9
                    outflows_cell_wise[key].append(float(listed[counter + 1]))

    for key in outflows_cell_wise:
        # print(key, outflows_cell_wise[key][:10])

        if plotting_enabled:
            plt.plot(time_stamps, outflows_cell_wise[key])
            plt.title("Cell number:" + str(key))
            plt.ylabel("Outflow")
            plt.xlabel(r"Time stamp: unit ($\Delta t$)")
            plt.grid(grid)
            plt.tight_layout()
            plt.savefig("output_images/flow_plots/Cell number:" + str(key) + ".png", dpi=DPI)
            plt.show(block=False)
            plt.close()

    # plot the moving average to show the average rate of arrival per time window
    for key in outflows_cell_wise:
        if plotting_enabled:
            flow_cumulative_in_last_x_minutes = np.convolve(
                outflows_cell_wise[key], np.ones(int(cumulative_window * 60)), mode="same"
            )
            plt.plot(time_stamps, flow_cumulative_in_last_x_minutes)
            plt.title("Cell number:" + str(key))
            plt.ylabel("Cumulative arrivals in the last " + str(cumulative_window) + " minutes")
            plt.xlabel(r"Time stamp: unit ($\Delta t$)")
            # plt.ylim(0, y_limit)
            plt.grid(grid)
            plt.tight_layout()
            plt.savefig(
                "output_images/flow_plots_cumulative/Cell number:"
                + str(key)
                + "_aggregation_x_"
                + str(cumulative_window)
                + ".png",
                dpi=DPI,
            )
            plt.show(block=False)
            plt.close()

    return outflows_cell_wise, time_stamps


def plot_vehicles_charging_currently(
    outflows_cell_wise,
    time_stamps,
    y_limit,
    grid,
    DPI=300,
    soc_sample_type=["uniform", 30],
    total_charging_time=90,
    max_vehicles=100,
):
    """

    :param outflows_cell_wise:  output from previous func process_outfl_file
    :param time_stamps:  output from previous func process_outfl_file
    :param soc_sample_type: ["uniform",  mean], or ["gaussian", mean, sigma]
    :param total_charging_time:
    :param cumulative_window:
    :param: max_vehicles:
    :param y_limit; y_limit,grid,DPi=300,; plotting related
    :return:
    """

    # plot the moving average to show the average rate of arrival per time window
    for key in outflows_cell_wise:
        if soc_sample_type[0] == "uniform":
            SOC = np.random.rand() * soc_sample_type[1]
        elif soc_sample_type[0] == "gaussian":
            mu, sigma = soc_sample_type[1, 2]
            SOC = np.random.normal(mu, sigma)
        else:
            print("Wrong type of SOC")
            return -1

        currently_charging_matrix_x_axis_time_y_axis_num_vehicles = (
            np.random.rand(max_vehicles, len(time_stamps) // 60) * 0
        )

        # in order to handle the size of the currently charging matrix we convert everything to minutes
        outflows_cell_wise_per_minute = {}
        time_stamps_min = []
        for key in outflows_cell_wise:
            outflows_cell_wise_per_minute[key] = [0] * (len(outflows_cell_wise[key]) // 60)
            for i in range(0, len(time_stamps), 60):
                t = time_stamps[i]
                t_min = t // 60
                time_stamps_min.append(t_min)
                outflows_cell_wise_per_minute[key][t_min - 1] = sum(outflows_cell_wise[key][t - 1 : t + 59])

        y_counter = 0
        for t in time_stamps_min:
            # edge overflow are already taken care of by numpy indexing
            # P.S it is forgiving for slices exceeding boundaries

            currently_charging_matrix_x_axis_time_y_axis_num_vehicles[
                y_counter : y_counter + int(outflows_cell_wise_per_minute[key][t - 1] * 100),
                t : int(t + total_charging_time - SOC),
            ] = 1
            y_counter += int(outflows_cell_wise_per_minute[key][t - 1] * 100)

        # column sum
        print(currently_charging_matrix_x_axis_time_y_axis_num_vehicles.shape)
        total_vehicles_charging_now = np.sum(currently_charging_matrix_x_axis_time_y_axis_num_vehicles, axis=0)

        plt.plot(time_stamps_min, total_vehicles_charging_now)
        plt.title("Cell number:" + str(key))
        plt.ylabel("outfl*100: just for reducing rounding error \n(Net effect scaling of image)")
        plt.xlabel("Time in minutes")
        # plt.ylim(0, y_limit)
        plt.grid(grid)
        plt.tight_layout()
        plt.savefig(
            "output_images/currently_Charging/Cell number:" + str(key) + "_aggregation_1_minute_" + ".png",
            dpi=DPI,
        )
        plt.show(block=False)
        plt.close()

        plt.imshow(currently_charging_matrix_x_axis_time_y_axis_num_vehicles, origin="lower", aspect="auto")
        plt.xlabel("Time in minutes")
        plt.ylabel("outfl*100: just for reducing rounding error \n(Net effect scaling of image)")
        plt.tight_layout()
        plt.savefig(
            "output_images/currently_Charging/Cell_number:" + str(key) + "_matrix.png",
            dpi=DPI,
        )
        plt.show(block=False)


def plot_cumulative_arrivals(
    x, peak_time, length_of_plot, plot_every_y_minutes, start_plotting, end_plotting, DPI, y_limit
):
    """

    Args:
        x: how many minutes window for cumulative arrival calculation

        peak_time : if we are using gaussian demand, when was the peak

        length_of_plot:

        start_plotting: as the name suggests; end_plotting : as the name suggests

        DPI, y_limit: as the name suggests

        Currently this reads from the hard coded path; TO-DO: need to change it to parameter

    Returns:
      outflows_cell_wise: per cell, per time step flow; cell_name is the key

    """

    flag_read_outflow = False  # initially, we set this to False

    time_stamps = []
    outflows_cell_wise = {}

    with open("input_output_text_files/outfl.txt") as f:
        for row in f:

            row_strip = row.strip()
            if "OUTFLOW START" in row_strip:
                flag_read_outflow = True

                for row_skip in f:
                    # using regex to split instead of standard split so that we can split based on any delimiter
                    # not necessarilty four spaces, tab etc..

                    # Also, we use 2:end because the first two are 'Time' and 'Step'
                    list_of_output_cells = [
                        int(x.replace("CELL", "")) for x in (list(filter(None, re.split("[, \-!?:]+", row_skip))))[2:]
                    ]
                    for key in list_of_output_cells:
                        outflows_cell_wise[key] = []
                    print(list_of_output_cells)
                    break
                next(f)
                continue

            if "OUTFLOW END" in row:
                flag_read_outflow = False

            if flag_read_outflow == True:
                # using regex to split instead of standard split so that we can split based on any delimiter
                # not necessarilty four spaces, tab etc..

                listed = list(filter(None, re.split("[, \-!?:]+", row_strip)))
                time_stamps.append(int(listed[0]))
                for counter, key in enumerate(list_of_output_cells):

                    # +1 because the first column of every row is the time stamp
                    if listed[counter + 1] == "******":
                        listed[counter + 1] = 10 ** 9
                    outflows_cell_wise[key].append(float(listed[counter + 1]))

    # plot the moving average to show the average rate of arrival per time window

    for key in outflows_cell_wise:

        color_counter = 0
        for plot_start_time in range(start_plotting, end_plotting, plot_every_y_minutes * 60):

            cumulative_flow_in_x_minutes = []
            for i in range(plot_start_time, plot_start_time + length_of_plot * 60):
                cumulative_flow_in_x_minutes.append(np.sum(outflows_cell_wise[key][i - x * 60 : i]))

            label = (plot_start_time - peak_time) // 60
            if label > 0:
                label = str(label) + " minutes after peak"
            elif label < 0:
                label = str(abs(label)) + " minutes before peak"
            else:
                label = "At peak"

            plt.plot(
                range(len(cumulative_flow_in_x_minutes)),
                cumulative_flow_in_x_minutes,
                label=label,
            )
            color_counter += 1

        plt.title("Cell number:" + str(key))
        plt.ylabel("Cumulative arrivals in the last " + str(x) + " minutes")
        plt.xlabel(r"Time stamp: unit ($\Delta t$) seconds ")
        # plt.ylim(0, y_limit)
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(
            "output_images/flow_plots_cumulative/Cell number:" + str(key) + "_x_val_" + str(x) + ".png", dpi=DPI
        )
        plt.show(block=False)
        plt.close()

    return outflows_cell_wise


#
# def generate_gephy_graph_with_flow_values(G_CTM, node_attributes, pos_nodes, outflows_cell_wise):
#     """
#      uses outflows_cell_wise to save dynamic gephi format; the dynamic axis being time and the
#      dynamic variable being flow
#
#     Args:
#
#       G: Takes the networkX graph, with attributes such as position, color etc.
#
#       outflows_cell_wise: output from the other function process_outfl_file()
#
#       node_attributes: same as the ones
#
#     """
#
#     mean_lat = []
#     mean_lon = []
#     for node in pos_nodes:
#         mean_lat.append(pos_nodes[node][0])
#         mean_lon.append(pos_nodes[node][1])
#     mean_lat = np.mean(mean_lat)
#     mean_lon = np.mean(mean_lon)
#
#     second_counter = 1
#
#
#     G_CTM_extended = nx.DiGraph(G_CTM)
#
#
#     time_attributes = {}
#     for second_counter in range(3600):
#
#         for node in G_CTM.nodes:
#
#             curent_datetime_format = dt.fromtimestamp(second_counter).strftime("%Y-%m-%d %H:%M:%S")
#
#             G_CTM_extended.add_node(node + second_counter)
#
#             time_attributes [node + second_counter] = {
#                 "start": second_counter ,
#                 "end": second_counter + 1 }
#
#             node_attributes[node + second_counter] = {
#                 "position": {
#                     "x": int((pos_nodes[node][0] - mean_lat) * 1000000),
#                     "y": int((pos_nodes[node][1] - mean_lon) * 1000000),
#                     "z": 0,
#                 },
#                 "label": node + second_counter,
#                 "start":node+second_counter
#             }
#
#             # with "label": node + second_counter, we are able to incorporate the name of the node
#             # along with the timestamp
#
#
#     nx.set_node_attributes(G_CTM_extended, node_attributes, "viz")
#     nx.set_node_attributes(G_CTM_extended, time_attributes, "attribute")
#     nx.write_gexf(G_CTM_extended, "output_images/network_graphs/Dynamic_CTM_after_run_complete.gexf")


def generate_gephy_graph_with_flow_values_as_video_frames(
    G_CTM,
    pos_nodes,
    outflows_cell_wise,
    startTime,
    endTime,
    print_every_n_time_steps=1,
    n_threads=-1,
    dpi=100,
    vmin=0,
    vmax=0.1,
    FPS=10,
):
    """
     uses outflows_cell_wise to save dynamic gephi format; the dynamic axis being time and the
     dynamic variable being flow

     Must be called with the ALL option, otherwise there is no point visualising part of the network as a
     video; if called without ALL, the following line will throw an error
     (outflows_cell_wise[node][second_counter])

    Args:

      G: Takes the networkX graph, with attributes such as position, color etc.

      outflows_cell_wise: output from the other function process_outfl_file()


      node_attributes: same as the ones

      startTime, endTime: actual simulation start and end time

      print_every_n_time_steps: as the name suggests

      n_threads : as the name suggests

      vmin, vmax: for color bar


    """

    mean_lat = []
    mean_lon = []
    for node in pos_nodes:
        mean_lat.append(pos_nodes[node][0])
        mean_lon.append(pos_nodes[node][1])
    mean_lat = np.mean(mean_lat)
    mean_lon = np.mean(mean_lon)

    for node in G_CTM.nodes:
        pos_nodes[node] = (
            (pos_nodes[node][0] - mean_lat) * 1000000,
            (pos_nodes[node][1] - mean_lon) * 1000000,
        )

    Parallel(n_jobs=n_threads)(
        delayed(save_graph_for_one_time_step)(G_CTM, outflows_cell_wise, pos_nodes, dpi, seconds_counter, vmin, vmax)
        for seconds_counter in list(range(startTime, endTime, print_every_n_time_steps))
    )


def save_graph_for_one_time_step(G_CTM, outflows_cell_wise, pos_nodes, dpi, second_counter, vmin, vmax):
    """
    vmin, vmax: for color bar
    Save graph for one time step; this function is parallelised
    """

    old_new_node_name_map = {}
    pos_nodes_for_this_time_step = {}

    for node in G_CTM.nodes:
        try:
            new_node_label = str(node) + "_" + str(outflows_cell_wise[node][second_counter])
        except KeyError:
            print("Key not found; Check if ALL nodes are being computed in outflows_cell_wise")
            sys.exit(0)

        old_new_node_name_map[node] = new_node_label
        pos_nodes_for_this_time_step[new_node_label] = pos_nodes[node]

    G_CTM_for_this_time_step = nx.DiGraph(G_CTM)
    G_CTM_for_this_time_step = nx.relabel_nodes(G_CTM_for_this_time_step, old_new_node_name_map)

    color_list = []
    for node in G_CTM.nodes:

        # we use a linear color scheme between "red" and "blue"
        # "blue" implying 0.1 and "red" implying 0
        # 255*10 because we thought the max value is 0.1 for the outfl.txt file

        color_list.append(outflows_cell_wise[node][second_counter])

    cmap = plt.cm.YlGnBu
    nx.draw(
        G_CTM_for_this_time_step,
        pos_nodes_for_this_time_step,
        connectionstyle="arc3, rad = 0.1",
        with_labels=False,
        node_size=3.3,
        node_color=color_list,
        width=0.8,
        arrowsize=2.6,
        node_shape="s",
        vmin=vmin,
        vmax=vmax,
        cmap=cmap,
    )

    plt.text(-1, -2, "Time step: " + str(second_counter), ha="right", va="top", fontsize=5)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=vmin, vmax=vmax))
    sm._A = []
    plt.colorbar(sm)

    # canvas = plt.gca().figure.canvas
    # canvas.draw()
    # data = np.frombuffer(canvas.tostring_rgb(), dtype=np.uint8)
    # return data.reshape(canvas.get_width_height()[::-1] + (3,))

    plt.grid(True)
    plt.show(block=False)
    # plt.tight_layout()
    plt.savefig("output_images/network_graphs/video/frame_" + f"{second_counter:09d}" + ".jpg", dpi=dpi)
    # plt.tight_layout()
    print(
        "Frames processed already: ",
        second_counter,
    )
    plt.close()


def save_csv_for_all_time_steps_kepler(
    G_CTM, outflows_cell_wise, pos_nodes, sim_end, scenario, cumulative_frames, arrows_only=True
):
    """

    :param G_CTM:
    :param outflows_cell_wise:
    :param pos_nodes:
    :param sim_end:
    :param scenario:
    :param cumulative_frames:
    :param arrows_only:
    :return:
    """
    if not arrows_only:
        with open("output_images/kepler_files/kepler-" + scenario + ".csv", "w") as f:
            csvwriter = csv.writer(f)
            csvwriter.writerow(["lat", "lon", "outflow", "time"])

            inDate = "29-Apr-2013-07:00:00"
            d = datetime.strptime(inDate, "%d-%b-%Y-%H:%M:%S")

            cum_val = {}
            for second_counter in range(sim_end):
                d += dt.timedelta(seconds=1)
                for node in G_CTM.nodes:

                    lon = pos_nodes[node][0]
                    lat = pos_nodes[node][1]

                    try:
                        outflow = outflows_cell_wise[node][second_counter]

                        if second_counter == 0:
                            cum_val[lat, lon] = outflow
                        else:
                            cum_val[lat, lon] += outflow

                    except:
                        print(
                            "Check: ALL cells are being processing in the outfl file\nIgnoring kepler file generation"
                        )
                        return

                    if second_counter % cumulative_frames == 0:
                        csvwriter.writerow([lat, lon, outflow, d.strftime("%Y/%`M/%DT%H:%M:%S")])
                        cum_val[lat, lon] = 0

                print(
                    "Frames processed already: ",
                    second_counter,
                )

    with open("output_images/kepler_files/kepler_gradient_arrows.csv", "w") as f:
        csvwriter = csv.writer(f)
        csvwriter.writerow(["start_lon", "start_lat", "end_lon", "end_lat", "node_id"])

        for edge in G_CTM.edges:
            start_node, end_node = edge

            start_lon = pos_nodes[start_node][0]
            start_lat = pos_nodes[start_node][1]
            end_lon = pos_nodes[end_node][0]
            end_lat = pos_nodes[end_node][1]

            csvwriter.writerow([start_lon, start_lat, end_lon, end_lat, start_node])


def convert_to_video():
    """
    converts the images inside network_graphs/video folder to a video for easy viewing
    """

    # choose codec according to format needed
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    file_list = glob.glob("output_images/network_graphs/video/*.jpg")

    # we sort it to keep the images in order of appearance
    list.sort(file_list)

    img = cv2.imread(file_list[0])
    width, height = img.shape[1], img.shape[0]
    video = cv2.VideoWriter("output_images/network_graphs/video/video.mp4", fourcc, 1, (width, height))

    for image_file in file_list:
        img = cv2.resize(cv2.imread(image_file), dsize=(width, height))
        video.write(img)

    cv2.destroyAllWindows()
    video.release()


if __name__ == "__main__":
    outflows_cell_wise = process_outfl_file()
