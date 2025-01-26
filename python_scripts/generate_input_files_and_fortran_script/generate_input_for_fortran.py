import networkx as nx
import sys
import warnings
from python_scripts.generate_input_files_and_fortran_script.generate_out30_for_fortran import generate_out_30_file
import shared_config

sys.path.append(shared_config.BASE_FOLDER_with_repo_name)

"""
98
1 3600
125
1-OR-1	0.6	1	1	OR	0	0	1-DI-3	0	0	0	0	N	0	0	0	1	35.208	13.3416	1783.673
1-OR-2	0.6	2	1	OR	0	0	1-TR-4	0	0	0	0	N	0	0	0	1	35.208	13.3416	1783.673
1-DI-3	0.6	1	1	DI	1-OR-1	0	1-TR-5	1-MB-6	0.88	0.12	0	N	0	0	0	1	35.208	13.3416	1783.673
1-TR-4	0.6	2	1	TR	1-OR-2	0	1-MB-7	0	0	0	0	N	0	0	0	1	35.208	13.3416	1783.673
1-TR-5	0.6	1	1	TR	1-DI-3	0	1-ME-8	0	0	0	0	Y	77	40	80	1	35.208	13.3416	1783.673
1-MB-6	0.1	1	1	MB	1-DI-3	0	3-ME-36	0	0	0	0	Y	77	40	80	1	35.208	13.3416	1783.673
.
.
.
7
1-OR-1	1	3600
1-OR-2	1	3600
2-OR-11	1	3600
2-OR-12	1	3600
3-OR-18	1	3600
3-OR-19	1	3600
1-OR-89 1	3600
120
"""


def generate_input_for_fortran(
    G,
    timeDelta,
    endTime,
    K_j,
    node_attributes,
    demand_type,
    gaussian_demand_max_val,
    y_lim_demand,
    file_name_split_ratio=None,
    real_params=None,
    seconds_offset_midnight=0,
    pos_nodes=None,
    dict_lat_lon_to_lanes=None,
    freeze_lanes_debug=None,
    smoothing_data_window=None,
    uniform_demand_value=0.1,
):
    """

    Args:
      G: param startTime:
      endTime: param K_j:
      startTime:
      K_j:
      node_attributes: keys are tghe nodes, values are in dict format (with keys color, position, label)
      demand_type : one of "uniform", "gaussian", "binary","two_peaks"
      gaussian_demand_max_val: as the name suggests
      y_lim_demand: for plotting

    Returns:
      None

    """
    if demand_type == "real_demand":
        if real_params == None:
            print("Wrong params for real demand mapping of sensors")
            sys.exit(0)

    G_copy = nx.DiGraph(G)

    print(G_copy.number_of_nodes())
    print(timeDelta, endTime)
    print(K_j)

    split_dict = {}
    if file_name_split_ratio != None:
        with open(file_name_split_ratio) as f:
            for row in f:
                #  format of row:  1376->1377:0.1|1421:0.9
                listed = row.strip().split("->")
                key = int(listed[0].strip())
                vals = listed[1].split("|")
                split_dict[key] = {
                    int(vals[0].split(":")[0]): float(vals[0].split(":")[1]),
                    int(vals[1].split(":")[0]): float(vals[1].split(":")[1]),
                }
                assert float(vals[0].split(":")[1]) + float(vals[1].split(":")[1]) == 1

    list_of_origin_cells = []
    for i in range(G_copy.number_of_nodes()):

        name_ = (list(G_copy.nodes))[i]  # node name same as actual node name in networkx

        node = name_  ## To:do; remove this; maybe this is not needed
        node_pred = list(G_copy.predecessors(node))
        node_succes = list(G_copy.successors(node))

        # we set all from and to zero; and later modify only the ones which matter
        from_1 = from_2 = to_1 = to_2 = 0

        type_ = node_attributes[node]["type"]

        if type_ == "OR":
            list_of_origin_cells.append(node)

        if len(node_pred) >= 1:
            from_1 = node_pred[0]
        if len(node_pred) == 2:
            from_2 = node_pred[1]

        if len(node_succes) >= 1:
            to_1 = node_succes[0]
        if len(node_succes) == 2:
            to_2 = node_succes[1]

        full_factor = 0
        turn_factor = 1

        if type_ == "DI":
            if pos_nodes is None:
                split_1 = 0.5
                split_2 = 0.5
                try:
                    if node in split_dict:
                        split_1 = split_dict[node][to_1]
                        split_2 = split_dict[node][to_2]
                except:
                    print("Something wrong with connection (to_1 or to_2 does not match split.txt)")
                    warnings.warn("Something wrong with connection (to_1 or to_2 does not match split.txt)")
                    print("Retaining 0.5 split")
                    continue
            else:
                lane_1 = dict_lat_lon_to_lanes[pos_nodes[node_succes[0]]]
                lane_2 = dict_lat_lon_to_lanes[pos_nodes[node_succes[1]]]
                sum = lane_1 + lane_2

                # rounding off to two decimal points is important, otherwise
                # fortran throws an error
                split_1 = round(lane_1 / sum, 2)
                split_2 = round(lane_2 / sum, 2)
        else:
            split_1 = 0
            split_2 = 0

        """
        NEED TO REVIEW THIS PART! 
        QUICKFIX to make all intersections free of red lights & hard coded priority to 0
        """
        ################################################## FROM HERE ################################
        if pos_nodes is None:
            if type_ == "ME":
                #     priority = 1
                # else:
                #     priority = 0
                priority = 0.5
            else:
                # this is dummy value, it is not used for non ME cells
                priority = 0.5
        else:
            lane_this_cell = dict_lat_lon_to_lanes[pos_nodes[node]]
            if freeze_lanes_debug is not None:
                lane_this_cell = freeze_lanes_debug

            if type_ == "ME":
                # max merges can be two
                assert len(list(G_copy.predecessors(node_succes[0]))) <= 2

                brother_cell = list(set(list(G_copy.predecessors(node_succes[0]))) - set([node]))[0]
                lane_brother_cell = dict_lat_lon_to_lanes[pos_nodes[brother_cell]]

                # rounding off to two decimal points is important, otherwise
                # fortran throws an error
                priority = round(lane_this_cell / (lane_brother_cell + lane_this_cell), 2)

            else:
                # this is dummy value, it is not used for non ME cells
                priority = 0.5

        if type_ == "MB":
            signal = "N"
        else:
            signal = "N"
        ################################################## UNTIL HERE ################################

        offset = 0
        green_effective = 30
        red_effective = 30
        f_max = 1
        f_flow = 97
        shockwave = 13
        sat_flow = 1783

        print_list = [
            name_,
            full_factor,
            lane_this_cell,
            turn_factor,
            type_,
            from_1,
            from_2,
            to_1,
            to_2,
            split_1,
            split_2,
            priority,
            signal,
            offset,
            green_effective,
            red_effective,
            f_max,
            f_flow,
            shockwave,
            sat_flow,
        ]

        print(*print_list, sep="\t")

    #  DEMAND RELATED
    print(len(list_of_origin_cells))
    for or_cell in list_of_origin_cells:
        print(*[or_cell, 1, endTime], sep="\t")

    if demand_type == "real_data":
        assert (
            generate_out_30_file(
                list_of_origin_cells,
                (endTime),
                demand_type,
                y_lim_demand,
                params=real_params,
                seconds_offset_midnight=seconds_offset_midnight,
            )
            == 0
        )

    else:
        assert (
            generate_out_30_file(
                list_of_origin_cells,
                (endTime),
                demand_type,
                y_lim_demand,
                params=[gaussian_demand_max_val],
                seconds_offset_midnight=seconds_offset_midnight,
                smoothing_data_window=smoothing_data_window,
                uniform_demand_value=uniform_demand_value,
            )
            == 0
        )

    # if everything goes well, return 0
    return 0


if __name__ == "__main__":
    generate_input_for_fortran()
