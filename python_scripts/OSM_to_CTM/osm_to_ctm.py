import networkx as nx
import osmnx as ox
import pandas as pd
import warnings
import sys
import csv
import numpy as np
from networkx_viewer import Viewer
import os
from shapely.geometry import LineString

import shared_config

sys.path.append(shared_config.BASE_FOLDER_with_repo_name)

from python_scripts.OSM_to_CTM.graphs_transforms import apply_three_transforms
from sklearn.neighbors import KDTree as KDT

# @if using networkx viewer, the backend needs to be set to Qt5
# import matplotlib
# matplotlib.use("Qt5Agg")

import matplotlib.pyplot as plt

# is NaN check using Chris-Jester Young's answer
def isNaN(num):
    return num != num


def fetch_road_network_from_osm_database(
    named_location=None,
    lat=None,
    lon=None,
    polygon=None,
    dist=1000,
    network_type="drive",
    custom_filter=None,
    simplify=True,
    missing_lane_default_value=-1,
):
    """

    Args:
      named_location: named location if we are using this instead of lat lon (Default value = None)
      lat: latitude of rctangle centre (Default value = None)
      lon: longitude of rectangle centre (Default value = None)
      polygon: polygon of handmade boundaries (Default value = None)
      dist: distance (not 100% what are the dimensions; visualise to make sure we have what we need) (Default value = 1000)
      network_type: drive/walking etc.. refer to osmnx readme@ https://osmnx.readthedocs.io/en/stable/osmnx.html#osmnx.graph.graph_from_bbox (Default value = "drive")
      custom_filter: highway"] @https://osmnx.readthedocs.io/en/stable/osmnx.html#osmnx.graph.graph_from_bbox
      simplify: as the name suggests



    Returns:
        G: Saves two graphs and returns one graph G, before splitting segments into cells split/diverge/merge transforms (Default value = None)
        dict_osm_id_to_lanes: as the name suggests

    """

    ox.config(use_cache=True, log_console=True)

    # download street network data from OSM and construct a MultiDiGraph model
    if lat != None and lon != None:
        G = ox.graph_from_point((lat, lon), dist=dist, network_type=network_type)
    elif named_location != None:
        G = ox.graph_from_address(
            address=named_location, dist=dist, network_type=network_type, custom_filter='["highway"~"motorway"]'
        )
    elif polygon != None:
        G = ox.graph_from_polygon(polygon, network_type=network_type, custom_filter=custom_filter, simplify=simplify)
    else:
        print("Error; wrong input \n\n\n")
        sys.exit()

    # impute edge (driving) speeds and calculate edge traversal times
    G = ox.add_edge_speeds(G)
    G = ox.add_edge_travel_times(G)
    G = ox.distance.add_edge_lengths(G)

    # add lane numbers
    edges = ox.graph_to_gdfs(G, nodes=False, edges=True)
    edges = edges.fillna(missing_lane_default_value)
    # edges["lanes"] = pd.to_numeric(edges["lanes"], downcast="float")
    dict_osm_id_to_lanes = {}
    for i in range(edges.shape[0]):
        u, v, key = edges.iloc[i].name
        dict_osm_id_to_lanes[u, v] = edges.iloc[i].lanes

    # edges_to_remove = []
    # for u in G.nodes:
    #     for v in G.nodes:
    #
    #         if G.has_edge(u, v):
    #             # edge exists
    #             e = G[u][v]
    #             highway_type = e[0]["highway"]
    #             if custom_filter is not None:
    #                 if highway_type not in custom_filter:  # motorway
    #                     edges_to_remove.append((u, v))
    #
    # G.remove_edges_from(edges_to_remove)

    # The nodes carry the default ids from osmnx, we convert them to sequential node numbers
    G = nx.convert_node_labels_to_integers(G, first_label=1, label_attribute="old_node_id")

    # we collect the coordinates (x,y) of the nodes manually
    pos_nodes = {}
    for u in G.nodes:
        pos_nodes[u] = (G.nodes[u]["x"], G.nodes[u]["y"])

    ox.plot_graph(
        G,
        bgcolor="k",
        node_size=50,
        edge_linewidth=2,
        edge_color="#333333",
        save=True,
        filepath="output_images/network_graphs/original_network.png",
        dpi=300,
        close=True,
        show=False,
    )

    # Plotting osm map with road names (labels misaligned)
    fig, ax = ox.plot_graph(G, bgcolor="k", edge_linewidth=3, node_size=0, show=False, close=False)
    for _, edge in ox.graph_to_gdfs(G, nodes=False).fillna("").iterrows():
        c = edge["geometry"].centroid
        text = edge["name"]
        ax.annotate(text, (c.x, c.y), c="w", fontsize=3, color="green")
    plt.tight_layout()
    plt.savefig("output_images/network_graphs/original_network_as_networkx_with_roadnames.png", dpi=300)
    plt.show(block=False)
    plt.close()

    nx.draw_networkx_nodes(G, pos_nodes, node_size=300)
    nx.draw(G, pos_nodes, connectionstyle="arc3, rad = 0.1", with_labels=True)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("output_images/network_graphs/original_network_as_networkx_with_directions.png", dpi=300)
    plt.show(block=False)
    plt.close()

    return G, dict_osm_id_to_lanes


def split_road_network_into_cells_based_on_speed(
    G,
    time_delta,
    call_type,
    default_speed,
    dict_osm_id_to_lanes,
    FORCE_SPEED="allow_original_speed",
    plot_nw_in_kepler=True,
):
    """

    Args:
      G: input networkx graph
      time_delta: splitting the network into cells (by using time quantum; To-Do: Clarificaiton needed); Units seconds
    In case where the maxspeed is not available, we revert to a value of default_speed
      call_type: either "using_dummy_data" or "using_osm"
      FORCE_SPEED: only for testing purpose

    Returns:
      G_CTM:
      color_nodes:
      pos_nodes:


    """

    OSM_cells = nx.DiGraph()
    OSM_edge_labels = {}

    counter_for_no_edge_between_u_v = 0
    counter_for_yes_edge_between_u_v = 0

    color_nodes = {}
    pos_nodes = {}
    shape_nodes = {}
    # color all orginal nodes red
    # collect the coordinates (x,y) of the nodes manually
    # mark the shape of original intersections as squares
    for u in G.nodes:
        color_nodes[u] = "red"
        pos_nodes[u] = (G.nodes[u]["x"], G.nodes[u]["y"])
        shape_nodes[u] = "o"

    dict_lat_lon_to_lanes = {}

    with open("output_images/kepler_files/kepler_lanes_speeds.csv", "w") as f:
        csvwriter = csv.writer(f)
        csvwriter.writerow(["lon", "lat", "number_of_lanes", "speed_limit_kph", "real/virtual"])

    for u in G.nodes:
        for v in G.nodes:

            if G.has_edge(u, v):
                # edge exists
                counter_for_yes_edge_between_u_v += 1
                e = G[u][v]
            else:
                # edge does not exist
                counter_for_no_edge_between_u_v += 1
                continue

            if call_type == "using_osm":
                length = e[0]["length"]
            elif call_type == "using_dummy_data":
                length = G[u][v]["length"]
            else:
                print(
                    "Error in call type for split_road_network_into_cells_based_on_speed() \n"
                    "It must be either 'using_osm' or 'using_dummy_data'"
                )
                sys.exit(0)

            try:
                speed_limit = float(e[0]["speed_kph"])
            except:
                speed_limit = default_speed
                warnings.warn("Maxspeed not found in json file for element ")
                print("Edge with missing maxspeed: ", "Edge between", u, "&", v, "\n\t maxspeed set as ", speed_limit)

            if FORCE_SPEED != "allow_original_speed":
                speed_limit = default_speed
                print("Edge between", u, "&", v, "\n\t maxspeed set as ", speed_limit)

            # based on the size of time delta
            cell_size = speed_limit * 5 / 18 * time_delta

            # TO-DO @Nishant, improve boundary conditions; This value must be at least one,
            # the least value is to ensure that when time delta is large, we have one entire segment being
            # considered as a single cell
            number_of_cells_in_this_segment = round(max(np.ceil(length / cell_size), 2))

            # Sometimes, geometry is missing, in that case, we draw a straight line
            if "geometry" in G[u][v][0]:
                segment_line_string = LineString(G[u][v][0]["geometry"])
            else:
                segment_line_string = LineString(
                    [(G.nodes[u]["x"], G.nodes[u]["y"]), (G.nodes[v]["x"], G.nodes[v]["y"])]
                )
                print("Warning!:   geometry missing for " + str(u) + "_" + str(v))
                print("Straight line filled in place of geometry")

            try:
                parent_number_of_lanes = int(dict_osm_id_to_lanes[G.nodes[u]["old_node_id"], G.nodes[v]["old_node_id"]])
            except:
                print("OSM graph not simplified, taking the maximum number of lanes from the list")
                print("This option is used if simplify=False when calling the OSM get map function")
                parent_number_of_lanes = int(
                    max(dict_osm_id_to_lanes[G.nodes[u]["old_node_id"], G.nodes[v]["old_node_id"]])
                )

            for small_edge_counter in range(1, number_of_cells_in_this_segment + 1):
                # @TO-DO: Ask Yi about rounding error;

                origin_node = str(u) + "_" + str(v) + "_" + str(small_edge_counter)
                dest_node = str(u) + "_" + str(v) + "_" + str(small_edge_counter + 1)

                # taking care of the boundary conditions
                if small_edge_counter == 1:
                    origin_node = u
                if small_edge_counter == number_of_cells_in_this_segment:
                    dest_node = v

                if origin_node == dest_node:
                    print("Something wrong, we must not have self loops!")
                    print("This is already taken care of by the test fiunction later on; # redundant")
                    sys.exit(0)

                OSM_cells.add_edge(origin_node, dest_node)

                # visualisation stuff
                # 1. position
                if origin_node not in pos_nodes:

                    # The output of .wkt is : 'POINT (-93.27498572354531 44.9353883315587)'
                    # some basic string processing using replace to extract the lat, lon from the return value
                    x, y = (
                        segment_line_string.interpolate(
                            small_edge_counter / (number_of_cells_in_this_segment + 2), normalized=True
                        )
                        .wkt.replace("POINT ", "")
                        .replace("(", "")
                        .replace(")", "")
                        .split(" ")
                    )
                    x = float(x)
                    y = float(y)
                    pos_nodes[origin_node] = (x, y)

                if dest_node not in pos_nodes:
                    x, y = (
                        segment_line_string.interpolate(
                            small_edge_counter / (number_of_cells_in_this_segment + 2), normalized=True
                        )
                        .wkt.replace("POINT ", "")
                        .replace("(", "")
                        .replace(")", "")
                        .split(" ")
                    )
                    x = float(x)
                    y = float(y)
                    pos_nodes[dest_node] = (x, y)

                with open("output_images/kepler_files/kepler_lanes_speeds.csv", "a") as f:
                    csvwriter = csv.writer(f)
                    csvwriter.writerow([x, y, parent_number_of_lanes, speed_limit, "real"])

                dict_lat_lon_to_lanes[x, y] = parent_number_of_lanes

                # 2. color
                if origin_node not in color_nodes:
                    color_nodes[origin_node] = "yellow"
                if dest_node not in color_nodes:
                    color_nodes[dest_node] = "yellow"

                # shape of node
                if origin_node not in shape_nodes:
                    shape_nodes[origin_node] = "s"
                if dest_node not in shape_nodes:
                    shape_nodes[dest_node] = "s"

                OSM_edge_labels[origin_node, dest_node] = str(u) + "_" + str(v) + "_" + str(small_edge_counter)

    color_list = []
    shape_list = []
    for node in OSM_cells.nodes:
        color_list.append(color_nodes[node])
        shape_list.append(shape_nodes[node])

    ensure_no_self_loops(OSM_cells)

    print("Total number of edges not present for UV pairs:", counter_for_no_edge_between_u_v)
    print("Total number of edges  present for UV pairs:", counter_for_yes_edge_between_u_v)

    nx.draw_networkx_nodes(
        OSM_cells,
        pos_nodes,
        node_size=20,
    )  # this part is ensure proper spacing, later on: as shown below we can edit colors/shapes etc at will
    nx.draw(
        OSM_cells,
        pos_nodes,
        node_size=20,
        connectionstyle="arc3, rad = 0.1",
        node_color=color_list,
        with_labels=True,
        font_size=6,
    )
    # if we want to mix different types of shape, we should use subgraphs as shown below:
    # we can keep track of shapes and use two subgraphs
    # OSM_cells.subgraph(list(OSM_cells.nodes)[10:]),

    plt.grid(True)
    plt.tight_layout()
    plt.savefig("output_images/network_graphs/networkx_split_into_CTM_cells.png", dpi=800)
    plt.show(block=False)
    plt.close()

    return OSM_cells, color_nodes, pos_nodes, dict_lat_lon_to_lanes


def remove_the_older_nodes(
    G, color_nodes, pos_nodes, save_intermediate_plots=True, uturns_allowed=False, using_nx_viewer=False
):
    """

    Args:
        G:
        color_nodes: dictionary of node id to colors
        pos_nodes: only used for visualisation
        save_intermediate_plots: If True, we save the intermediate graphs while
        removing the intersection nodes one by one
        uturns_allowed: If False, no u turns allowed
        using_nx_viewer: as the name suggests

    Returns:
        pos_nodes:

    """

    G_copy = nx.DiGraph(G)

    for u in G.nodes:
        if color_nodes[u] == "red":

            pred = list(G.predecessors(u))
            succ = list(G.successors(u))
            grandPa_to_grandChildren = [(o, d) for o in pred for d in succ]

            # u-turns allowed?
            if not uturns_allowed:

                uturns_to_remove = []
                for od_pair in grandPa_to_grandChildren:
                    # # we have named the cells as "<origin>_<dest>_<cell_num>"
                    # # If there is a u turn, the CTM node naming must be symmetric
                    # # symmetric implying Origin for one direction should be destination for the other one
                    if (
                        od_pair[1].split("_")[0] == od_pair[0].split("_")[1]
                        and od_pair[0].split("_")[0] == od_pair[1].split("_")[1]
                    ):
                        uturns_to_remove.append(od_pair)

                grandPa_to_grandChildren = list(set(grandPa_to_grandChildren) - set(uturns_to_remove))

            G_copy.add_edges_from(grandPa_to_grandChildren)
            G_copy.remove_node(u)

            if save_intermediate_plots:
                nx.draw_networkx_nodes(
                    G_copy,
                    pos_nodes,
                    node_size=20,
                )  # this part is ensure proper spacing, later on: as shown below we can edit colors/shapes etc at will
                nx.draw(
                    G_copy,
                    pos_nodes,
                    node_size=20,
                    node_shape="s",
                    connectionstyle="arc3, rad = 0.1",
                    node_color="yellow",
                    with_labels=True,
                    font_size=6,
                )
                plt.grid(True)
                plt.tight_layout()
                plt.savefig(
                    "output_images/network_graphs/removing_intersection_nodes/CTM_cells_original_"
                    + str(u)
                    + "_removed.png",
                    dpi=300,
                )
                plt.show(block=False)
                plt.close()

    # plotting with actual cell locations
    nx.draw_networkx_nodes(
        G_copy,
        pos_nodes,
        node_size=20,
    )  # this part is ensure proper spacing, later on: as shown below we can edit colors/shapes etc at will
    nx.draw(
        G_copy,
        pos_nodes,
        node_size=20,
        node_shape="s",
        connectionstyle="arc3, rad = 0.1",
        node_color="yellow",
        with_labels=True,
        font_size=6,
    )
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("output_images/network_graphs/CTM_cells_red_nodes_removed_UT_" + str(uturns_allowed) + ".png", dpi=800)
    plt.show(block=False)
    plt.close()

    # The nodes carry the default ids from osmnx, we convert them to sequential node numbers
    # ordering = default is important to ensure that we can refer to the older names for plotting (position)
    old_names = list(G_copy.nodes)
    G_copy = nx.convert_node_labels_to_integers(G_copy, first_label=1, ordering="default")
    new_names = list(G_copy.nodes)

    # get older_pos with new names of nodes
    for j in range(len(old_names)):
        pos_nodes[new_names[j]] = pos_nodes[old_names[j]]

    # plotting the graph with node colors by degree
    degrees = G_copy.degree()  # Dict with Node ID, Degree
    node_colors = np.asarray([degrees[n] for n in G_copy.nodes()])

    # plotting with actual cell locations
    nx.draw_networkx_nodes(
        G_copy,
        pos_nodes,
        node_size=20,
    )  # this part is ensure proper spacing, later on: as shown below we can edit colors/shapes etc at will
    nx.draw(
        G_copy,
        pos_nodes,
        node_size=20,
        node_shape="s",
        connectionstyle="arc3, rad = 0.1",
        node_color=node_colors,
        with_labels=True,
        font_size=6,
    )
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("output_images/network_graphs/CTM_cells_final_sequential_numbers.png", dpi=800)
    plt.show(block=False)
    plt.close()

    node_attributes = {}

    # compute the centroid of all the node positions
    mean_lat = []
    mean_lon = []
    for key in pos_nodes:
        mean_lat.append(pos_nodes[key][0])
        mean_lon.append(pos_nodes[key][1])
    mean_lat = np.mean(mean_lat)
    mean_lon = np.mean(mean_lon)

    for key in pos_nodes:
        # some quickfix 🧑‍🔧 solutions applied here to increase the initial spacing of nodes
        # because the default value of lat, lon has very small differences  😂
        # and we end up with a big lump of nodes instead of a graph!!
        node_attributes[key] = {
            "color": {"a": 0.6, "r": 0, "b": 0, "g": 0},
            "position": {
                "x": int((pos_nodes[key][0] - mean_lat) * 1000000),
                "y": int((pos_nodes[key][1] - mean_lon) * 1000000),
                "z": 0,
            },
            "label": key,
        }
    nx.set_node_attributes(G_copy, node_attributes, "viz")
    nx.write_gexf(G_copy, "output_images/network_graphs/gephi.gexf")

    # True option not working now; 😿 😢 for now, we do this using gephi
    if using_nx_viewer:
        app = Viewer(G_copy)
        app.mainloop()

    return G_copy, pos_nodes


def retain_only_the_largest_components(G, pos_nodes):
    """

    Args:
        G:
        pos_nodes: just for plotting the graph

    Returns:
        G_copy:
        max_component_size:

    """
    G_copy = nx.DiGraph(G)
    print("\n\n\n\n")

    components_size_list = []

    # the graph is converted to undirected for the purposes of determining
    # connnected components; networkx rule; connected components is not defined
    # for undirected components
    for comp in list(nx.connected_components(nx.Graph(G_copy))):
        components_size_list.append(len(comp))
    max_component_size = max(components_size_list)

    # we remove all other components of the graph
    for comp in list(nx.connected_components(nx.Graph(G_copy))):
        if len(comp) < max_component_size:
            for u in comp:
                G_copy.remove_node(u)

    nx.draw_networkx_nodes(
        G_copy,
        pos_nodes,
        node_size=20,
    )  # this part is ensure proper spacing, later on: as shown below we can edit colors/shapes etc at will
    nx.draw(
        G_copy,
        pos_nodes,
        node_size=20,
        node_shape="s",
        connectionstyle="arc3, rad = 0.1",
        node_color="yellow",
        with_labels=True,
        font_size=6,
    )
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(
        "output_images/network_graphs/retaining_the_largest_component.png",
        dpi=300,
    )

    plt.show(block=False)
    plt.close()

    # save same thing without labels to get a clearer picture
    nx.draw_networkx_nodes(
        G_copy,
        pos_nodes,
        node_size=20,
    )  # this part is ensure proper spacing, later on: as shown below we can edit colors/shapes etc at will
    nx.draw(
        G_copy,
        pos_nodes,
        node_size=5,
        node_shape="s",
        connectionstyle="arc3, rad = 0.1",
        node_color="blue",
        with_labels=False,
        font_size=6,
    )
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(
        "output_images/network_graphs/retaining_the_largest_component_no_labels.png",
        dpi=300,
    )
    plt.show(block=False)
    plt.close()

    print("\n\n\n\n")
    print(len(list(G.nodes)) - len(list(G_copy.nodes)), " nodes removed when removing the smaller components")
    print("\n\n\n\n")

    return G_copy, max_component_size


def isIntStr(n):
    # first, the item must be a string
    if not isinstance(n, str):
        return False

    # second, we try to convert it to integer
    try:
        int(n)
        return True
    except:
        return False

def reduce_node_degree_to_ensure_valid_CTM(G, pos_nodes, dict_lat_lon_to_lanes):
    """

    Args:
        G:
        pos_nodes

    Returns:
        G_copy: contains some nodes edited from G
        dummy_count: positions of nodes in G_copy
        pos_nodes:
        color_nodes:
    """

    plot_degree_dist(G, "before_node_split_")

    all_positions_list_of_list = []
    lat_list = []
    lon_list = []
    for key in pos_nodes:
        all_positions_list_of_list.append([pos_nodes[key][0], pos_nodes[key][1]])
        lat_list.append(pos_nodes[key][0])
        lon_list.append(pos_nodes[key][1])

    # convert all to numpy arrays
    all_positions_list_of_list = np.array(all_positions_list_of_list)
    lat_list = np.array(lat_list)
    lon_list = np.array(lon_list)

    tree = KDT(all_positions_list_of_list, leaf_size=2, metric="euclidean")

    G_copy, dummy_count, parent_cell_pointer = apply_three_transforms(
        G,
        dummy_counter=1,
    )

    color_nodes = {}

    for u in G_copy.nodes:

        if u not in pos_nodes:

            # implies u is a dummy
            assert "dummy" in u

            list_of_neighbours = list(G_copy.neighbors(u))

            # This function must be called after the smaller components are removed;
            # so here, we must not have any any nodes with no neighbours
            assert len(list_of_neighbours) >= 1

            # we will set the dummy node location to some random location within its nearest n_neighbours neighbours
            n_neighbours = 5
            corresponding_original_node = u.split("_")[0]
            _, ind = tree.query(
                X=(np.array(pos_nodes[int(corresponding_original_node)])).reshape(-1, 2), k=n_neighbours
            )

            rand_lat = np.random.rand() * (np.max(lat_list[ind]) - np.min(lat_list[ind])) + np.min(lat_list[ind])
            rand_lon = np.random.rand() * (np.max(lon_list[ind]) - np.min(lon_list[ind])) + np.min(lon_list[ind])

            # sometime for complicated intersections,
            # we have a dummy node as the parents of a dummy node;
            # hence we need to query twice so that we finally get an integer as the parent
            # integer? Why? because we have the real nodes marked as integers when we
            # first assign integers to the OSM nodes and let go of the OSM node ids.
            parent_cell = u
            while not isIntStr(parent_cell):
                parent_cell = parent_cell_pointer[parent_cell]

            with open("output_images/kepler_files/kepler_lanes.csv", "a") as f3:
                csvwriter_3 = csv.writer(f3)
                try:
                    csvwriter_3.writerow(
                        [rand_lat, rand_lon, dict_lat_lon_to_lanes[pos_nodes[int(parent_cell)]], "virtual"]
                    )
                except KeyError:
                    print ("Keyerrror while mapping new nodes to old ones/ writing the kepler file")
                    debug_stop = True
                    sys.exit(0)

            dict_lat_lon_to_lanes[rand_lat, rand_lon] = dict_lat_lon_to_lanes[pos_nodes[int(parent_cell)]]

            # add this new position to the dictionary
            pos_nodes[u] = (rand_lat, rand_lon)

            color_nodes[u] = "black"

        else:
            color_nodes[u] = "yellow"

    color_list = []
    for node in G_copy.nodes:
        color_list.append(color_nodes[node])

    nx.draw_networkx_nodes(G_copy, pos_nodes, node_size=5)
    nx.draw(
        G_copy,
        pos_nodes,
        node_size=5,
        node_shape="s",
        connectionstyle="arc3, rad = 0.1",
        node_color=color_list,
        with_labels=True,
        font_size=6,
    )
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(
        "output_images/network_graphs/final_CTM.png",
        dpi=300,
    )
    plt.show(block=False)
    plt.close()
    plot_degree_dist(G_copy, "after_node_split_")

    #################
    # saving in GEPHI
    #################
    node_attributes = {}
    # compute the centroid of all the node positions
    mean_lat = []
    mean_lon = []
    for key in G_copy.nodes:
        mean_lat.append(pos_nodes[key][0])
        mean_lon.append(pos_nodes[key][1])
    mean_lat = np.mean(mean_lat)
    mean_lon = np.mean(mean_lon)

    for key in G_copy.nodes:
        # some quickfix 🧑‍🔧 solutions applied here to increase the initial spacing of nodes
        # because the default value of lat, lon has very small differences  😂
        # and we end up with a big lump of nodes instead of a graph!!
        if color_nodes[key] == "black":
            color_nodes[key] = {"a": 0.6, "r": 0, "b": 0, "g": 0}
        elif color_nodes[key] == "yellow":
            color_nodes[key] = {"a": 0.6, "r": 255, "b": 0, "g": 255}

        node_attributes[key] = {
            "color": color_nodes[key],
            "position": {
                "x": int((pos_nodes[key][0] - mean_lat) * 1000000),
                "y": int((pos_nodes[key][1] - mean_lon) * 1000000),
                "z": 0,
            },
            "label": key,
        }
    nx.set_node_attributes(G_copy, node_attributes, "viz")
    nx.write_gexf(G_copy, "output_images/network_graphs/CTM_after_node_split.gexf")

    # we make sure that the in degree and the out degrees are within the CTM limits
    # in other words, we check if this function did it's job
    in_degrees = G_copy.in_degree()  # Dict with Node ID, Degree
    out_degrees = G_copy.out_degree()  # Dict with Node ID, Degree
    degree_in = np.asarray([in_degrees[n] for n in G_copy.nodes()])
    degree_out = np.asarray([out_degrees[n] for n in G_copy.nodes()])

    # This test is also commented out for now
    assert np.max(degree_out) == 2 and np.max(degree_in) == 2

    return G_copy, pos_nodes, color_nodes, dict_lat_lon_to_lanes


def plot_degree_dist(G, fname):
    """

    Args:
        G:
        fname:

    Returns:
        None; plots the degree distribution
    """
    in_degrees = G.in_degree()  # Dict with Node ID, Degree
    out_degrees = G.out_degree()  # Dict with Node ID, Degree
    degree_in = np.asarray([in_degrees[n] for n in G.nodes()])
    plt.hist(degree_in)
    plt.title("Histogram of degree distribution")
    plt.savefig("output_images/network_graphs/" + fname + "CTM_final_degree_distribution_in_deg.png", dpi=300)
    plt.show(block=False)
    plt.close()

    degree_out = np.asarray([out_degrees[n] for n in G.nodes()])
    plt.hist(degree_out)
    plt.title("Histogram of degree distribution")
    plt.savefig("output_images/network_graphs/" + fname + "CTM_final_degree_distribution_out_deg.png", dpi=300)
    plt.show(block=False)
    plt.close()

    degree_total = np.asarray([in_degrees[n] + out_degrees[n] for n in G.nodes()])
    plt.hist(degree_total)
    plt.title("Histogram of degree distribution")
    plt.savefig("output_images/network_graphs/" + fname + "CTM_final_degree_distribution_total_deg.png", dpi=300)
    plt.show(block=False)
    plt.close()


def ensure_no_self_loops(G):
    """
    tests if the OSM cells have self loops, should not have
    Args:
        u:
        v:
        number_of_parts:

    Returns:

    """
    for u in G.nodes:
        assert not G.has_edge(u, u)
    print("No self loops found")


def assign_cell_type_as_an_attribute(G_CTM, pos_nodes, create_kepler_file=False):
    """

    Args:
        G_CTM: CTM graph
        pos_nodes: dictionary of node positions

    Returns:
        G_CTM_copy :
        node_attributes: including color, position, labels
    """

    node_attributes = {}
    color_based_on_node_type = {}
    color_based_on_node_type["DE"] = {"a": 0.6, "r": 255, "b": 0, "g": 0}
    color_based_on_node_type["OR"] = {"a": 0.6, "r": 0, "b": 0, "g": 255}
    color_based_on_node_type["ME"] = {"a": 0.6, "r": 255, "b": 255, "g": 255}
    color_based_on_node_type["MB"] = {"a": 0.6, "r": 255, "b": 255, "g": 255}
    color_based_on_node_type["DI"] = {"a": 0.6, "r": 255, "b": 255, "g": 255}
    color_based_on_node_type["TR"] = {"a": 0.6, "r": 255, "b": 255, "g": 255}

    # The nodes carry some "_dummy" names in labels, we convert them to sequential node numbers
    # ordering = default is important to ensure that we can refer to the older names for plotting (position)
    G_CTM_copy = G_CTM
    old_names = list(G_CTM_copy.nodes)
    G_CTM_copy = nx.convert_node_labels_to_integers(G_CTM_copy, first_label=1, ordering="default")
    new_names = list(G_CTM_copy.nodes)

    # get older_pos with new  of nodes
    for j in range(len(old_names)):
        pos_nodes[new_names[j]] = pos_nodes[old_names[j]]

    mean_lat = []
    mean_lon = []
    for node in pos_nodes:
        mean_lat.append(pos_nodes[node][0])
        mean_lon.append(pos_nodes[node][1])
    mean_lat = np.mean(mean_lat)
    mean_lon = np.mean(mean_lon)

    if create_kepler_file:
        with open("output_images/kepler_files/kepler_node_types.csv", "w") as f:
            csvwriter = csv.writer(f)
            csvwriter.writerow(["lon", "lat", "node_type", "cell_num"])

    for node in G_CTM_copy.nodes:
        node_pred = list(G_CTM_copy.predecessors(node))
        node_succes = list(G_CTM_copy.successors(node))

        """
        Method for node type assignment:
        0. Default: All are set to TR 
        1.  We fix OR and DE type
        2. We fix DI type
        3. We fix ME based on the following
        4. To-do: @Nishant, difference between ME and MB we will fix later
        """
        type_ = "TR"
        if len(node_pred) == 0:
            type_ = "OR"
        elif len(node_succes) == 0:
            type_ = "DE"
        elif (len(node_pred) == 1 and len(node_succes) == 2) or (len(node_pred) == 2 and len(node_succes) == 2):
            type_ = "DI"
        elif len(node_succes) == 1:
            # we check if the current cell leads to a cell where more cells lead to
            predecessors_of_successor = list(G_CTM_copy.predecessors(node_succes[0]))
            if len(predecessors_of_successor) > 1:
                type_ = "ME"

        node_attributes[node] = {
            "color": color_based_on_node_type[type_],
            "position": {
                "x": int((pos_nodes[node][0] - mean_lat) * 100000),
                "y": int((pos_nodes[node][1] - mean_lon) * 100000),
                "z": 0,
            },
            "label": node,
            "type": type_,
            "size": get_node_size(type_),
        }

        if create_kepler_file:
            with open("output_images/kepler_files/kepler_node_types.csv", "a") as f2:
                csvwriter = csv.writer(f2)
                csvwriter.writerow([pos_nodes[node][0], pos_nodes[node][1], type_, node])


    df = pd.read_csv('output_images/kepler_files/kepler_node_types.csv')  # make sure to provide the correct path to your CSV file
    df['selected_labels'] = df.apply(lambda row: str(row['cell_num']) if row['node_type'] in ['OR', 'DE'] else '', axis=1)
    df.to_csv('output_images/kepler_files/kepler_node_types_selected_labels.csv', index=False)  # index=False ensures that pandas does not write row numbers                                     axis=1)
    print("New CSV file with 'selected_labels' column has been created as 'new_data.csv'.")

    nx.set_node_attributes(G_CTM_copy, node_attributes, "viz")
    nx.write_gexf(G_CTM_copy, "output_images/network_graphs/CTM_after_node_type_assignment.gexf")

    return G_CTM_copy, node_attributes, pos_nodes


def get_node_size(type_):
    """
    OR/DE: big size (say 40)
    ALl else, small size (say 10)
    :param type_:
    :return: None
    """
    if type_ in ["OR", "DE"]:
        return 40
    else:
        return 10
