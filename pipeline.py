import os
import sys
import contextlib
import warnings
import shared_config

sys.path.append(shared_config.BASE_FOLDER_with_repo_name)

import pandas as pd
from slugify import slugify

import python_scripts.generate_input_files_and_fortran_script.generate_input_for_fortran as generate_input_for_fortran
import python_scripts.visualisations.plot_arrivals as plot_arrivals
import shared_config
from python_scripts.generate_input_files_and_fortran_script.generate_scenarios_config import get_scenario_config

# from python_scripts.optimise_charging_schedule.EV_charging_ACN import optimise_charging_EV
from python_scripts.process_real_data.raw_data_new_format.visualise_sensor_locations import (
    generate_sensor_locations_file_from_poly,
)
from python_scripts.visualisations.plot_arrivals import create_combined_destination_cells
import csv
import python_scripts.visualisations.visualise_CTM_connections as visualise_CTM_connections
from python_scripts.generate_input_files_and_fortran_script.generate_accident_file import (
    generate_single_accident,
    generate_single_accident_with_variable_flow,
)
import matplotlib.pyplot as plt
import numpy as np
import pickle
import time
from python_scripts.visualisations.plot_arrivals import (
    generate_gephy_graph_with_flow_values_as_video_frames,
    save_csv_for_all_time_steps_kepler,
    convert_to_video,
)
from shapely.geometry import Polygon
import shared_config


from python_scripts.process_real_data import raw_signal_split_into_modes

### CHECK speed limit units 5/18 or NOT in osm_networks.py

overall_run_start_time = time.time()

# chdir step is important important step to ensure the working directory is reset after all those imports
# we have several of those imports to ensure that the scripts can be run independently
# by taking care of the relative position of the folders for inputs and outputs
from python_scripts.OSM_to_CTM.osm_to_ctm import (
    fetch_road_network_from_osm_database,
    split_road_network_into_cells_based_on_speed,
    remove_the_older_nodes,
    reduce_node_degree_to_ensure_valid_CTM,
    assign_cell_type_as_an_attribute,
    retain_only_the_largest_components,
)
from smartprint import smartprint as sprint
from python_scripts.process_real_data.raw_signal_split_into_modes import raw_data_to_one_d_signals_minnst

os.chdir("./")


### INPUT PARAMETERS
END_TIME = shared_config.END_TIME
CELL_TYPE_TO_ANALYSE = "ALL"  # Maybe ALL, DE, etc. Use None with caution, see doc in genereate_fortran_code_outflow.py
DEMAND_TYPE = "real_data"
# DEMAND_TYPE = "uniform"
# SIMULATING_OR_PLOTTING = "DEBUG_WITH_MULTIPLE_UNIFORM_DEMANDS"
SIMULATING_OR_PLOTTING = "SPP"
stochasticity_in_demand = False
# run_counter = 0; now this is being set inside the function
TIME_DELTA = 1
PLOTTING_KEPLER_OUTFLOW_VIDEO = False
PLOT_NETWORK_CELLS = True
SECONDS_OFFSET_MIDNIGHT = shared_config.SECONDS_OFFSET_MIDNIGHT
ADDING_LENGTH_OF_WINDOW_IN_SCENARIO_PLOTS = False
DESTINATIONS_HARDCODED = shared_config.DESTINATIONS_HARDCODED
    # [
    # 1601,
    # 1600,
    # 1581,
    # 1580,
    # 1576,
    # 1575,
    # 1646,
    # 1645,
    # "1601+1581+1576+1646",
    #
    # 6156,
    # 6220,
    # 2360,
    # 3964,

    # 2276,
    # 2347,
    # 2407,
    # 2386,2933,2944,2557,2172,995,1329,
    # 3908, 3573, 3873, 4028, 4126,
    # 6136, 6131, 6200, 3703, 4046

    # 6156, 6200,
    # 4028, 3873,
    # 1152,
    # 2933, 2557, 2407, 2386, 2276,
    # 1329


# ]
MAKE_SMALL_PICKLE_FILE = True


def reset_folder_structure():
    os.system("cp -r output_images output_images_backup_" + str(int(np.random.rand() * 1000000)))
    os.system("rm -rf output_images")
    os.system("mkdir output_images")
    os.system("mkdir output_images/network_graphs")
    os.system("mkdir output_images/network_graphs/video")
    os.system("mkdir output_images/flow_plots")
    os.system("mkdir output_images/split_merge_transforms")
    os.system("mkdir output_images/flow_plots_moving_average")
    os.system("mkdir output_images/flow_plots_cumulative")
    os.system("mkdir output_images/currently_Charging")
    os.system("mkdir output_images/scenario_plots_combined")
    os.system("mkdir output_images/raw_data")
    os.system("mkdir output_images/kepler_files")

    ## clear old folders
    #


SCENARIO_PARAMS = get_scenario_config("default", SECONDS_OFFSET_MIDNIGHT, frac=0.5)

subset_of_scenarios = {}
if shared_config.list_of_scenarios != []:
    # implies we run all scenarios
    for key in SCENARIO_PARAMS:
        if key in shared_config.list_of_scenarios:  # "No accident",
            subset_of_scenarios[key] = SCENARIO_PARAMS[key]
    SCENARIO_PARAMS = subset_of_scenarios


# @Singapore
# G_OSM = osm.split_road_network_into_cells(lat=1.292730, lon=103.768155, dist=300, time_delta=1)
# G = ox.graph_from_point((40.764, -73.985), dist=150, network_type="drive")  # @Manhattan
# lat=36.61659, lon=-100.82891;  @ Some highway in the US; custom filter ["highway"]


def simulating(uniform_demand_value=0.1):
    """G_OSM = fetch_road_network_from_osm_database(
        lat=44.978329342549515, lon=-93.27733142838234,
        dist=3000,
        network_type="drive",
        custom_filter=["highway", "motorway"],
    )
    """

    reset_folder_structure()

    fname = "input_output_text_files/G_OSM.pkl"
    # no need to query from the api everytime
    if not os.path.exists(fname):
        # if the new graph is being generated, we should delete the intermediate pickle files as well
        os.system("rm input_output_text_files/G_CTM_and_other_files.pkl")

        # geo = {
        #     "type": "Polygon",
        #     "coordinates": [
        #         [
        #             [-93.28388214111328, 44.96759138429636],
        #             [-93.28285217285156, 44.961882876810925],
        #             [-93.2819938659668, 44.896498217861584],
        #             [-93.26401233673096, 44.895981407050456],
        #             [-93.2680892944336, 44.9452399436188],
        #             [-93.25916290283203, 44.95714559892792],
        #             [-93.25881958007812, 44.96236872935042],
        #             [-93.26139450073242, 44.97184203149381],
        #             [-93.28182220458984, 44.97147770264882],
        #             [-93.28388214111328, 44.96759138429636],
        #         ]
        #     ],
        # }

        geo = {
            "type": "Polygon",
            "coordinates": shared_config.polygon_coordinates,
        }

        poly = Polygon([tuple(l) for l in geo["coordinates"][0]])

        # generate kepler file to show sensor locations
        generate_sensor_locations_file_from_poly(poly)

        # G_proj = osm.project_graph(G)
        # fig, ax = osm.plot_graph(G_proj)
        # , "trunk","trunk_link", "motorway_link","primary","secondary"]
        # custom_filter=["motorway", "motorway_link","motorway_junction","highway"],

        G_OSM, dict_osm_id_to_num_lanes = fetch_road_network_from_osm_database(
            polygon=poly,
            network_type="drive",
            custom_filter='["highway"~"motorway|motorway_link|primary"]',
            simplify=True,
            missing_lane_default_value=1,
        )
        with open(fname, "wb") as handle:
            pickle.dump((G_OSM, dict_osm_id_to_num_lanes), handle, protocol=4)
    else:
        with open(fname, "rb") as handle:
            G_OSM, dict_osm_id_to_num_lanes = pickle.load(handle)

    fname = "input_output_text_files/G_CTM_and_other_files.pkl"
    # no need to query from the api everytime
    if not os.path.exists(fname):
        #'["highway"~"motorway|motorway_link|primary|trunk"]'
        # ["highway"~"motorway"]
        os.system("rm output_images/network_graphs/kepler_lanes_virtual_cells.csv")
        with open("output_images/network_graphs/kepler_lanes.csv", "a") as f:
            csvwriter = csv.writer(f)
            csvwriter.writerow(["lon", "lat", "lanes", "type"])

        G_CTM, color_nodes, pos_nodes, dict_lat_lon_to_lanes = split_road_network_into_cells_based_on_speed(
            G_OSM,
            time_delta=TIME_DELTA,
            call_type="using_osm",
            default_speed=97,
            FORCE_SPEED="default_speed",
            dict_osm_id_to_lanes=dict_osm_id_to_num_lanes,
        )

        G_CTM_red_removed, pos_nodes = remove_the_older_nodes(
            G_CTM,
            color_nodes,
            pos_nodes,
            save_intermediate_plots=False,
            uturns_allowed=False,
            using_nx_viewer=False,
        )

        G_CTM_smaller_components_removed, _ = retain_only_the_largest_components(G_CTM_red_removed, pos_nodes)

        G, pos_nodes, color_nodes, dict_lat_lon_to_lanes = reduce_node_degree_to_ensure_valid_CTM(
            G_CTM_smaller_components_removed, pos_nodes, dict_lat_lon_to_lanes
        )

        G_valid_CTM, node_attributes, pos_nodes = assign_cell_type_as_an_attribute(
            G, pos_nodes=pos_nodes, create_kepler_file=True
        )
        with open(fname, "wb") as handle:
            pickle.dump(
                (G_valid_CTM, node_attributes, pos_nodes, dict_lat_lon_to_lanes, dict_osm_id_to_num_lanes),
                handle,
                protocol=4,
            )
    else:
        with open(fname, "rb") as handle:
            G_valid_CTM, node_attributes, pos_nodes, dict_lat_lon_to_lanes, dict_osm_id_to_num_lanes = pickle.load(
                handle
            )

    outflows_cell_wise_scenario = {}  # to save the results across scenarios for common plotting

    if PLOT_NETWORK_CELLS:
        ###### PLOT THE NETWORK
        # Initialize lists to hold plot data
        x_coords = []
        y_coords = []
        colors = []
        sizes = []
        ids = []
        # Extract data from node_attributes
        for node, attrs in node_attributes.items():
            x_coords.append(attrs['position']['x'])
            y_coords.append(attrs['position']['y'])
            # Convert color from RGBA to a matplotlib color
            rgba_color = (
            attrs['color']['r'] / 255, attrs['color']['g'] / 255, attrs['color']['b'] / 255, attrs['color']['a'])

            attr_size = attrs['size']
            # making them visible in white background; reduce their size
            if rgba_color[0] == 1 and rgba_color[1] == 1 and rgba_color[2] == 1:
                rgba_color = (0, 0, 0, 0.1)
                # print (rgba_color)
                attr_size = 1

            colors.append(rgba_color)
            sizes.append(attr_size)
            ids.append(attrs["label"])

        # Create scatter plot
        plt.figure(figsize=(10, 6))
        scatter = plt.scatter(x_coords, y_coords, c=colors, s=sizes, alpha=0.6)

        plt.xlabel('X Coordinate')
        plt.ylabel('Y Coordinate')
        plt.title('Node Plot')
        plt.grid(True)
        plt.colorbar(scatter, label='Node Size')
        plt.show()

        with open("visualise_Cells_in_kepler.csv", "w") as f:
            def rgba_to_hex(r, g, b, a=1.0):
                """Convert RGBA color to a hexadecimal format, assuming alpha is between 0 and 1."""
                return '#{:02x}{:02x}{:02x}{:02x}'.format(r, g, b, int(a * 255))

            csvwriter = csv.writer(f)
            csvwriter.writerow(["x", "y", "color", "size", "ID"])
            for i, _ in enumerate(x_coords):
                csvwriter.writerow([x_coords[i], y_coords[i], colors[i], sizes[i], ids[i]])

    run_counter = 0
    for SCENARIO in SCENARIO_PARAMS:
        # SCENARIO = "No accident"
        # if SCENARIO == "No accident":

        ## generate input for fortran
        ## no need to regenerate the input again if there is no stochasticity across multiple scenarios
        ## however, we must do it at least once (for the first input)
        ## this block needs to be modified if we have the demand loaded from real data,
        if stochasticity_in_demand or (not stochasticity_in_demand and run_counter == 0):
            run_counter += 1

            with open("input_output_text_files/input.txt", "w") as f:
                with contextlib.redirect_stdout(f):
                    return_code = generate_input_for_fortran.generate_input_for_fortran(
                        G=G_valid_CTM,
                        timeDelta=1,
                        endTime=END_TIME,
                        K_j=125,
                        node_attributes=node_attributes,
                        demand_type=DEMAND_TYPE,
                        uniform_demand_value=uniform_demand_value,
                        gaussian_demand_max_val=2,
                        y_lim_demand=3,
                        file_name_split_ratio=None,  # used to override the split ratio from lane %, no longer needed
                        real_params=[
                            shared_config.CTM_flow_data_file_pickle,
                            shared_config.sensor_cell_num_map,
                        ],
                        seconds_offset_midnight=SECONDS_OFFSET_MIDNIGHT,
                        pos_nodes=pos_nodes,
                        dict_lat_lon_to_lanes=dict_lat_lon_to_lanes,
                        freeze_lanes_debug=5,
                        smoothing_data_window=5,
                    )
                    if return_code != 0:
                        if return_code == -1:
                            warnings.warn("CTM cells not defined properly; some cells do not have a type")
                            print("CTM cells not defined properly; some cells do not have a type")
                        else:
                            warnings.warn("Some unknown error with CTM")
                            print("Some unknown error with CTM")
                        sys.exit(0)

        ### generate accident file
        if (
            SCENARIO_PARAMS[SCENARIO]["accident_duration_start_time_delta_unit"]
            != SCENARIO_PARAMS[SCENARIO]["accident_duration_end_time_delta_unit"]
        ):
            # accident case
            list_of_accident_cells = list(SCENARIO_PARAMS[SCENARIO]["dict_key_cell_value_fractional_capacity"].keys()) # [0]

        else:
            list_of_accident_cells = []  # no accident case

        generate_single_accident_with_variable_flow(
            list_of_all_cells=list_of_accident_cells,  # list(range(1, len(list(G_valid_CTM.nodes)) + 1)),
            dict_key_cell_value_fractional_capacity=SCENARIO_PARAMS[SCENARIO][
                "dict_key_cell_value_fractional_capacity"
            ],
            accident_duration_start_time_delta_unit=SCENARIO_PARAMS[SCENARIO][
                "accident_duration_start_time_delta_unit"
            ],
            accident_duration_end_time_delta_unit=SCENARIO_PARAMS[SCENARIO]["accident_duration_end_time_delta_unit"],
            file_path="input_output_text_files/accident.txt",
            list_of_flow_vals=SCENARIO_PARAMS[SCENARIO]["list_of_flows"],
        )

        ## generate svg file for CTM visualisation
        # visualise_CTM_connections.visualise_CTM_connections_from_input_file()

        ## generate code for fortran
        command = (
            "python python_scripts/generate_input_files_and_fortran_script/generate_fortran_code_for_outflow.py --starttime "
            + str(SCENARIO_PARAMS[SCENARIO]["accident_duration_start_time_delta_unit"])
            + " --endtime "
            + str(SCENARIO_PARAMS[SCENARIO]["accident_duration_end_time_delta_unit"])
            + " --which_type_of_cells "
            + CELL_TYPE_TO_ANALYSE
            + " > fortran_codes_and_exec/main_modified.f90"
        )
        print("Command to run generate fortran code: ", command)
        generate_fortran_code = os.system(command)

        if generate_fortran_code == 0:
            print("Fortran code generated successfully!\n\n")
        elif generate_fortran_code == -1:
            print("Fortran code generation crashed!!!!\n Error code: -1 : No destination cells \n\n")
            sys.exit(0)
        else:
            print("Fortran code generation crashed!!!!\n Error code: ", generate_fortran_code, "\n\n")

        # compile fortran code
        compile_fortran_code = os.system(
            " export LIBRARY_PATH=:/Library/Developer/CommandLineTools/SDKs/MacOSX.sdk/usr/lib && gfortran -ffree-line-length-none -ffpe-trap=zero,invalid,overflow,underflow fortran_codes_and_exec/main_modified.f90 -o fortran_codes_and_exec/a.out"
        )
        if compile_fortran_code == 0:
            print("Fortran compiled successfully!\n\n")
        else:
            print("Fortran compilation crashed!!!!\nError code: ", compile_fortran_code)
            sys.exit(0)

        # run fortran code
        startime = time.time()
        run_fortran_code = os.system("fortran_codes_and_exec/a.out")
        if run_fortran_code == 0:
            print("Fortran ran successfully!\n")
            print("CTM simulation time: ", round(time.time() - startime, 2), "seconds")
        else:
            print("Fortran run crashed!!!!\n Error code: ", run_fortran_code)
            sys.exit(0)

        for cumulative_window in [90]:
            #  no need to have multiple cumulative windows here;
            # we will be redoing them after all the scenarios anyway
            # the outflows_cell_wise remains the same across different cumulative windows

            outflows_cell_wise, time_stamps = plot_arrivals.process_outfl_file(
                plotting_enabled=False, cumulative_window=cumulative_window, DPI=300, y_limit=240, grid=False
            )

            outflows_cell_wise = create_combined_destination_cells(
                outflows_cell_wise, destination_merge_file_path="input_output_text_files/destinations_to_merge.txt"
            )
            outflows_cell_wise_scenario[SCENARIO] = outflows_cell_wise, time_stamps

        if PLOTTING_KEPLER_OUTFLOW_VIDEO:
            save_csv_for_all_time_steps_kepler(
                G_valid_CTM,
                outflows_cell_wise,
                pos_nodes,
                sim_end=END_TIME,
                scenario=SCENARIO,
                cumulative_frames=60,
                arrows_only=True,
            )

        # plot_arrivals.plot_vehicles_charging_currently(
        #     outflows_cell_wise,
        #     time_stamps,
        #     y_limit=240,
        #     DPI=300,
        #     grid=True,
        #     soc_sample_type=["uniform", 30],
        #     total_charging_time=90,
        #     max_vehicles=550000,
        # )

        # for x_val in [5, 10, 15, 20, 25]:
        #     plot_cumulative_arrivals = plot_arrivals.plot_cumulative_arrivals(
        #         x=x_val,
        #         peak_time=3600,
        #         length_of_plot=30,
        #         plot_every_y_minutes=60,
        #         DPI=300,
        #         start_plotting=1800,
        #         end_plotting=5000,
        #     )

        # plot_arrivals.generate_gephy_graph_with_flow_values_as_video_frames(
        #     G_valid_CTM,
        #     pos_nodes,
        #     outflows_cell_wise,
        #     startTime=1,
        #     endTime=END_TIME,
        #     print_every_n_time_steps=1,
        #     n_threads=6,
        #     dpi=100,
        #     vmin=0,
        #     vmax=0.1,
        #     FPS=10,
        # )
        #
        # plot_arrivals.convert_to_video()

        os.system("rm -rf output_images_" + SCENARIO.replace(" ", "\\ "))
        os.system("cp -r output_images output_images_" + SCENARIO.replace(" ", "\\ "))

        # pickle file generated inside the loop,
        # so that the outputs are usable even if one of the later
        # scenarios crashes

        if MAKE_SMALL_PICKLE_FILE:
            to_delete = []
            for scenario in outflows_cell_wise_scenario:
                for cell_ in outflows_cell_wise_scenario[scenario][0]:
                    if shared_config.OUTFLOW_ALL_DESTINATION_CELLS:
                        aa = []
                        for key in node_attributes:
                            if node_attributes[key]["type"] == "DE":
                                aa.append(key)
                        DESTINATIONS_HARDCODED = aa
                    if cell_ not in DESTINATIONS_HARDCODED:
                        # reduce the size of pickle files by saving only what is needed
                        # for power side simulation
                        #  this will not work for cases of combined destinations in the  string formatl
                        # in that case, MAKE_SMALL_PICKLE_FILE must be false
                        # If we really want to make that work for small pickle file, we need to change the
                        # function create_combined_destination_cells above to return the timestamps as well
                        try:
                            to_delete.append((scenario, cell_))

                        except Exception:
                            print("Error in deleting selected cells")
                            sprint(scenario, cell_)
                            debug_here = True
                            sys.exit(0)

            # Other loop so that the dict size doesn't change during iteration
            for key in to_delete:
                scenario, cell_ = key
                del outflows_cell_wise_scenario[scenario][0][cell_]
                del outflows_cell_wise_scenario[scenario][1][cell_]

        # else:
        with open("pickle_file_outflow_dicts.pickle", "wb") as handle:
            pickle.dump(outflows_cell_wise_scenario, handle, protocol=pickle.HIGHEST_PROTOCOL)

        # generate_gephy_graph_with_flow_values_as_video_frames(
        #     G_CTM=G_valid_CTM,
        #     pos_nodes=pos_nodes,
        #     outflows_cell_wise=outflows_cell_wise_scenario["No accident"][0],
        #     startTime=1,
        #     endTime=3600 * 3,
        #     print_every_n_time_steps=10,
        #     n_threads=7,
        #     dpi=300,
        #     vmin=0,
        #     vmax=0.1,
        #     FPS=10,
        # )
        # convert_to_video()


def helper_bin_into_gran(a, gran):
    assert isinstance(a, list)
    len_ = len(a)
    temp = []
    for i in range(0, len_, gran):
        try:
            temp.append(int(np.sum(a[i : i + gran]) * shared_config.ev_penetration))
        except:
            sprint([i, i + gran, shared_config.ev_penetration, np.sum(a[i : i + gran]), gran])
            print("Error in helper_bin_into_gran")
            sys.exit(0)
    return temp


def convert_seconds_list_to_5_min_list_int(a, cutoff_time=288):
    a = helper_bin_into_gran(a, 5 * 60)
    a = np.array(a)
    a[cutoff_time:] = 0
    a = a.tolist()
    return a


def convert_seconds_list_to_1_min_list_int(a):
    return helper_bin_into_gran(a, 60)


def plotting(uniform_demand_value=0.1):

    # Combining plots for scenarios
    with open("pickle_file_outflow_dicts.pickle", "rb") as handle:
        outflows_cell_wise_scenario = pickle.load(handle)

    enter_arrival_optim_file_creation = 0
    scenario_plotting = {}

    os.system("rm input_output_text_files/arrivals_for_optimiser" +  shared_config.prefix + "*")
    for SCENARIO in SCENARIO_PARAMS:
        ev_arrivals_file_name = "input_output_text_files/arrivals_for_optimiser" +  shared_config.prefix + "" + slugify(SCENARIO) + ".csv"
        with open(ev_arrivals_file_name, "w") as f3:
            csvwriter = csv.writer(f3)
            csvwriter.writerow(["cell", "SCENARIO", "outflow"])

        for cell in DESTINATIONS_HARDCODED:  # ["1601+1581+1576+1646"]:  # DESTINATIONS_HARDCODED:
            for cumulative_window in [1 / 60, 5]:  # [1/60, 1, 30, 90, 120]:  # [1, 30, 60, 90, 120]:

                # read back the dictionary and timestamp for the current scenario
                outflows_cell_wise = outflows_cell_wise_scenario[SCENARIO][0]

                # time_stamps = outflows_cell_wise_scenario[SCENARIO][1]
                # time_stamps is no longer needed in the DS. This will be removed in a future commit from the pickle
                # file
                try:
                    time_stamps = list(range(len(outflows_cell_wise[cell])))
                except KeyError:
                    print (KeyError) # debug_stop here
                    raise (KeyError)


                # np.zeros(int(cumulative_window * 60) term ensures that the convolution is shifted
                flow_cumulative_in_last_x_minutes = np.convolve(
                    outflows_cell_wise[cell],
                    np.concatenate([np.zeros(int(cumulative_window * 60)), np.ones(int(cumulative_window * 60))]),
                    mode="same",
                )

                if len(time_stamps) != len(flow_cumulative_in_last_x_minutes):
                    warnings.warn(
                        "Error in Cell:"
                        + str(cell)
                        + " scenario:"
                        + SCENARIO
                        + " cumulative_window: "
                        + str(cumulative_window)
                        + "\nLength of timestamps and the values do not match!!"
                    )
                    print("Check: Length of convolve kernel (filter) more than length of the array to be operated on")
                    print("Error in Cell:", cell, " scenario:", SCENARIO, " cumulative_window: ", cumulative_window)

                # diff = len(time_stamps) - len(flow_cumulative_in_last_x_minutes)
                # if diff % 2 != 0:
                #     # some boiler plate code to ensure that the edge effects are taken care of
                #     flow_cumulative_in_last_x_minutes = flow_cumulative_in_last_x_minutes[:-1]
                #     diff = diff // 2 * 2  # convert to even
                # time_stamps[diff // 2 + 1 : -diff // 2 - 1],

                sprint(cell, cumulative_window)
                if cumulative_window == 5: #  and cell == 6156:  # 1601:  # "1601+1581+1576+1646":

                    # This if-else block (isinstance(outflows_cell_wise[cell], list):) might not be needed anymore
                    # earlier we had a mix of numpy and lists, hence this was needed.
                    if not isinstance(outflows_cell_wise[cell], list):
                        outflow_list = outflows_cell_wise[cell].tolist()
                    else:
                        outflow_list = outflows_cell_wise[cell]

                    with open(ev_arrivals_file_name, "a") as f2:
                        csvwriter = csv.writer(f2)
                        csvwriter.writerow(
                            [cell, slugify(SCENARIO)]
                            + convert_seconds_list_to_5_min_list_int(outflow_list, cutoff_time=288)
                        )
                    enter_arrival_optim_file_creation += 1

                    scenario_plotting[cell, SCENARIO] = (
                        time_stamps[1800:],
                        flow_cumulative_in_last_x_minutes[1800:],
                    )

                # color=SCENARIO_PARAMS[SCENARIO]["color"],
                print("Total flow: (Conservation check)", np.sum(outflows_cell_wise[cell]))
                print("Cell: ", cell)

                if SCENARIO_PARAMS[SCENARIO]["dict_key_cell_value_fractional_capacity"] != {}:

                    # accident_marker = np.random.rand( len(time_stamps)) * 0
                    # accident_marker[
                    #     SCENARIO_PARAMS[SCENARIO]["accident_duration_start_time_delta_unit"]:
                    #     SCENARIO_PARAMS[SCENARIO]["accident_duration_end_time_delta_unit"],
                    # ] = 10
                    #
                    # #  we don't want to plot a line for the case when there was no accident
                    # # workaround trick: we make it np.nan; these are not plotted in matplotlib
                    # accident_marker[ accident_marker==0 ] = np.nan
                    #
                    # print(accident_marker.flatten())

                    if ADDING_LENGTH_OF_WINDOW_IN_SCENARIO_PLOTS:
                        plt.axvspan(
                            SCENARIO_PARAMS[SCENARIO]["accident_duration_start_time_delta_unit"],
                            SCENARIO_PARAMS[SCENARIO]["accident_duration_end_time_delta_unit"],
                            alpha=0.3,
                            color="gray",
                        )
                        # plt.hlines(
                        #     xmin=SCENARIO_PARAMS[SCENARIO]["accident_duration_start_time_delta_unit"],
                        #     xmax=SCENARIO_PARAMS[SCENARIO]["accident_duration_start_time_delta_unit"]
                        #     + int(cumulative_window * 60),
                        #     y=11,
                        # )
                        # plt.text(
                        #     x=SCENARIO_PARAMS[SCENARIO]["accident_duration_start_time_delta_unit"],
                        #     y=10,
                        #     s="Length of cumulative window",
                        #     fontsize=5,
                        # )

                    # SCENARIO_PARAMS[SCENARIO]["color"]

    # test to ensure that number of new arrival files is same as number of
    # try:
    #     assert enter_arrival_optim_file_creation == len(SCENARIO_PARAMS)
    # except Exception as e:
    #     raise e # debug here

    for key in scenario_plotting:
        cell, SCENARIO = key
        time_stamps, flow_cumulative_in_last_x_minutes = scenario_plotting[key]
        plt.plot(
            time_stamps,
            flow_cumulative_in_last_x_minutes,
            label="Scenario: " + SCENARIO + " Cell: " + str(cell),
            alpha=0.5,
            linewidth=2,
            color=SCENARIO_PARAMS[SCENARIO]["color"],
        )
        # plt.title("Cell number:" + str(cell))
        # plt.title(
        #     "Effect of accidents with varying accident duration reduction\n (Number of cells affected, capacity reduction & Start time of accidents remains same)",
        #     fontsize=6,
        # )

    plt.ylabel("Cumulative arrivals in the last " + str(cumulative_window) + " minutes", fontsize=11)
    plt.xlabel("Seconds since midnight", fontsize=11)
    plt.legend(loc="lower right", fontsize=9)
    plt.grid(alpha=0.3)
    # plt.ylim(0, 2800)
    plt.tight_layout()

    sprint(os.getcwd())
    # os.system("mkdir output_images")
    # os.system("mkdir output_images/scenario_plots_combined")
    plt.savefig(
        "output_images/scenario_plots_combined.png",
        dpi=300,
    )
    plt.close()


sprint(SCENARIO_PARAMS)

if SIMULATING_OR_PLOTTING in ["S", "SIMULATING"]:
    simulating()
elif SIMULATING_OR_PLOTTING in ["PLOTTING", "P"]:
    plotting()
elif SIMULATING_OR_PLOTTING in ["SIMULATE_AND_PLOT", "SP"]:
    simulating()
    plotting()
elif SIMULATING_OR_PLOTTING in ["SIMULATE_PLOT_AND_RUN_POWER", "SPP"]:
    os.system("rm input_output_text_files/arrivals_for_optimiser" +  shared_config.prefix + "*")
    os.system("rm -rf output_images*")
    os.system("mkdir output_images")
    if shared_config.CTM_SIMULATION_ENABLED:
        simulating()
    plotting()
    # os.system("conda activate congestion-and-grid-overload")

    if shared_config.POWER_OPTIM_ENABLED:
        os.system("rm python_scripts/optimise_charging_schedule/events_dict_across_traffic_scenarios.pickle")
        os.system("rm python_scripts/optimise_charging_schedule/*.png")
        os.system("rm -rf python_scripts/optimise_charging_schedule/results*")
        os.system("rm -rf python_scripts/optimise_charging_schedule/overall_power_side_results*.csv")

        for SCENARIO in SCENARIO_PARAMS:

            os.system("rm python_scripts/optimise_charging_schedule/events_dict_across_traffic_scenarios.pickle")

            command = (
                "python python_scripts/optimise_charging_schedule/compare_without_dollars_organised.py --scenario "
                + slugify(SCENARIO)
                + " --inputfilename arrivals_for_optimiser" +  shared_config.prefix + ""
                + slugify(SCENARIO)
                + ".csv"
            )
            sprint(command)
            os.system(command)

            command = (
                "mv python_scripts/optimise_charging_schedule/results python_scripts/optimise_charging_schedule/results-"
                + slugify(SCENARIO)
            )
            os.system("rm -rf python_scripts/optimise_charging_schedule/results-" + slugify(SCENARIO))
            os.system(command)

            pass
else:
    print("Wrong input in SIMULATING_OR_PLOTTING")
    print("Exiting code! :( ")
    sys.exit(0)

print("Total run time of main using time.time(): ", round(time.time() - overall_run_start_time, 2), " seconds")

# print ("Analysing the combined output file for power side results")
#
# # Read the CSV without headers
# df = pd.read_csv("python_scripts/optimise_charging_schedule/overall_power_side_results.csv", header=None, skiprows=1)
#
# # List of known column names
# known_cols = [
#     "Traffic-scenario",
#     "Transformer-capacity",
#     "Scenario",
#     "Algorithm",
#     "Run_number",
#     "proportion_delivered",
#     "demands_fully_met",
#     "peak_current",
#     "demand_charge",
#     "energy_cost",
#     "total_energy_delivered",
#     "total_energy_requested",
#     "aggregate_power_total",
#     "num_time_steps"
# ]
#
# # Compute the number of tod_ columns based on the remaining columns in the dataframe
# num_tod_cols = df.shape[1] - len(known_cols)
# # Create tod_ column names
# tod_cols = [f"tod_{i+1}" for i in range(num_tod_cols)]
# # Set new column names
# df.columns = known_cols + tod_cols
# print(df)
# # Grouping by the specified columns
# grouped = df.groupby(['Traffic-scenario', 'Transformer-capacity', 'Scenario', 'Algorithm']).mean().reset_index()
# print(grouped)
#
# grouped.to_csv("python_scripts/optimise_charging_schedule/overall_power_side_mean_results.csv", index=False)
#
# # mean_df.to_csv("python_scripts/optimise_charging_schedule/overall_power_side_mean_results.csv", index=False)