import networkx as nx
import pydot
import os
import sys

import shared_config

sys.path.append(shared_config.BASE_FOLDER_with_repo_name)

connections_file = "input_output_text_files/input.txt"  # To-do: @Nishant make this as a parameter
node_colors = {"TR": "black", "ME": "red", "DI": "blue", "DE": "red", "OR": "green"}

os.system("rm output_images/network_graphs/CTM_cells.svg")


def visualise_CTM_connections_from_input_file():
    """saves CTM connections plot in svg
    :return: None, saves the plot with rectangular cells
    colors the cells as well

    Args:

    Returns:

    """
    graph = pydot.Dot("my_graph", graph_type="digraph", bgcolor=None)

    # # Or, without using an intermediate variable:
    # graph.add_node(pydot.Node("b", shape="box"))

    # dictionary of cell id (string) to integer (cell number)
    cell_id_to_integer = {}
    counter = 1

    list_of_node_labels = []
    list_of_cell_names = []

    with open(connections_file) as f:
        for row in f:
            listed = row.strip().split("\t")

            # To-do:Change this later on
            if len(listed) < 10:
                # currently this is done so that the cell_connections.txt is not needed separately
                # we can parse the input.txt itself
                continue

            cell_name = listed[0]
            if cell_name not in cell_id_to_integer:
                cell_id_to_integer[cell_name] = counter

                # Keep track of these new nodes
                list_of_node_labels.append(str(cell_id_to_integer[cell_name]))
                list_of_cell_names.append(cell_name)
                counter += 1

    cell_id_to_integer["0"] = "dummy"  # special case to handle 0's in the
    # input.txt (for example Merge cell has 3 zeros, merge cell has 1 zero and so on )

    existing_edges = set([])  #  This will be used to keep track of existing edges
    # so that we do not have double edges

    with open(connections_file) as f:
        for row in f:
            listed = row.strip().split("\t")

            # To-do:Change this later on
            if len(listed) < 10:
                # this is done so that the cell_connections.txt is not needed separately
                # we can parse the input.txt itself
                continue

            cell_name = listed[0]

            # here origin and destination refer to the preceding and following cells of the CTM cell in consideration
            origin_1, origin_2, destination_1, destination_2 = listed[5:9]
            # Add edges, total of 4 edges can be added at max;

            # -------Plus we filter out the edges which have been added already
            if origin_1 != "0" and (origin_1, cell_name) not in existing_edges:
                new_edge = pydot.Edge(origin_1, cell_name, color="#f58610")
                graph.add_edge(new_edge)
                existing_edges.add((origin_1, cell_name))

            if origin_2 != "0" and (origin_2, cell_name) not in existing_edges:
                new_edge = pydot.Edge(origin_2, cell_name, color="#f58610")
                graph.add_edge(new_edge)
                existing_edges.add((origin_2, cell_name))

            if destination_1 != "0" and (cell_name, destination_1) not in existing_edges:
                new_edge = pydot.Edge(cell_name, destination_1, color="#f58610")
                graph.add_edge(new_edge)
                existing_edges.add((cell_name, destination_1))

            if destination_2 != "0" and (cell_name, destination_2) not in existing_edges:
                new_edge = pydot.Edge(cell_name, destination_2, color="#f58610")
                graph.add_edge(new_edge)
                existing_edges.add((cell_name, destination_2))

    #  convert to networkx from pydot
    graph_nx = nx.nx_pydot.from_pydot(graph)

    node_colors_dict = {}
    for node in graph_nx.nodes:

        node_pred = list(graph_nx.predecessors(node))
        node_succes = list(graph_nx.successors(node))

        if len(node_pred) > 2:
            print("More than two predecesssors found; Invalid CTM cell")
            return False

        if len(node_succes) > 2:
            print("More than two successors found; Invalid CTM cell")
            return False

        # we set all to zero; and later modify only the ones which matter
        type_ = "undefined"

        if len(node_pred) == 2 and len(node_succes) == 1:
            type_ = "ME"

        elif (len(node_pred) == 1 and len(node_succes) == 2) or (len(node_pred) == 2 and len(node_succes) == 2):
            type_ = "DI"

        elif len(node_pred) == 1 and len(node_succes) == 1:
            type_ = "TR"

        elif len(node_pred) == 0:
            type_ = "OR"

        elif len(node_succes) == 0:
            type_ = "DE"

        node_colors_dict[node] = node_colors[type_]

    # convert back to pydot for displaying
    graph = nx.nx_pydot.to_pydot(graph_nx)

    for cell_name in list_of_cell_names:
        # To-do @Nishant: creating a new Node works; not sure if this is the right way
        # the behavior is to replace the old nodes with new;
        # Maybe we can put a test here for asserting that the edges do not change while replacing nodes;

        new_node = pydot.Node(
            cell_name,
            label=str(cell_id_to_integer[cell_name]),
            shape="box",
            fontcolor=node_colors_dict[cell_name],
            color=node_colors_dict[cell_name],
        )
        graph.add_node(new_node)

    graph.write_svg("output_images/network_graphs/CTM_cells.svg", prog="dot")


# for png output
# graph.sa("output.png")


if __name__ == "__main__":
    visualise_CTM_connections_from_input_file()
