import networkx as nx

import sys

import shared_config

sys.path.append(shared_config.BASE_FOLDER_with_repo_name)
#########################################################
################  MERGING    ########################
#########################################################
def split_merging(G, parent_cell_pointer, dummy_counter=1):
    """

    Args:
      G: as the name suggests
      dummy_counter: as the name suggests

    Returns:
      G with each merging node > 2 incoming split into several consecutive nodes
      and dummy_counter

    """

    # we need two copies; one to ensure the sanctity of the input G
    # and second, to ensure that while we change the Graph in the loop,
    # the loop doesn't go crazy due to changing bounds

    G_copy = nx.DiGraph(G)
    G_copy_2 = nx.DiGraph(G)

    for node in G_copy.nodes:
        in_deg = G_copy.in_degree[node]
        if in_deg > 2:  # node must be split for incoming

            new_nodes = [str(node) + "_dummy" + str(i) for i in range(dummy_counter, dummy_counter + in_deg - 2)]

            for new_node in new_nodes:
                parent_cell_pointer[new_node] = str(node)

            dummy_counter = dummy_counter + in_deg - 2

            upstreams = [i for i in G_copy_2.predecessors(node)]
            downstreams = [i for i in G_copy_2.successors(node)]

            for up in upstreams:
                G_copy_2.remove_edge(up, node)

            for down in downstreams:
                G_copy_2.remove_edge(node, down)

            prev_node = node
            G_copy_2.add_edge(upstreams[0], prev_node)
            G_copy_2.add_edge(upstreams[1], prev_node)

            for i in range(2, len(upstreams)):
                G_copy_2.add_edge(prev_node, new_nodes[i - 2])
                G_copy_2.add_edge(upstreams[i], new_nodes[i - 2])
                prev_node = new_nodes[i - 2]

            for down in downstreams:
                G_copy_2.add_edge(prev_node, down)

    return G_copy_2, dummy_counter, parent_cell_pointer


#########################################################
################   DIVERGING    ########################
#########################################################
def split_diverging(G, parent_cell_pointer, dummy_counter):
    """

    Args:
      G: param dummy_counter: as  the name suggests
      dummy_counter:

    Returns:
      G with each diverging node > 2 outgoing split into several consecutive nodes
      and dummy_counter

    """

    G_copy = nx.DiGraph(G)

    # Trick: reverse; split; reverse
    G_copy = G_copy.reverse()

    G_copy, dummy_counter, parent_cell_pointer = split_merging(G_copy, parent_cell_pointer, dummy_counter=dummy_counter)
    G_copy = G_copy.reverse()

    return G_copy, dummy_counter, parent_cell_pointer


######################################################################
############  SPECIAL_CASE_OF_CELL MARKED AS ME and DI   #############
######################################################################
def remove_ME_DI_overlap(G, parent_cell_pointer, dummy_counter):
    """

    Args:
      G: input graph
      dummy_counter: as the name suggests

    Returns:
      check if there is any node that is DI but also qualifies as ME,
      we split them with an additional dummy node

    """

    # we need two copies; one to ensure the sanctity of the input G
    # and second, to ensure that while we change the Graph in the loop,
    # the loop doesn't go crazy due to changing bounds

    G_copy = nx.DiGraph(G)
    G_copy_2 = nx.DiGraph(G)

    for node in G_copy.nodes:
        out_deg = G_copy.out_degree[node]
        if out_deg >= 1:  # qualifies as DI

            # now we check if this can also be ME
            # How do we check for ME?
            # If the downstream cell has more than one incoming
            # then this is ME; so, we add additional node and limit the degree to 1

            downstreams = [i for i in G_copy.successors(node)]
            if len(downstreams) > 1:

                # we add this additional assert because the graph degree has already been
                # constrained to 2
                assert len(downstreams) == 2

                # out of the two downstreams, we retain just one edge and
                # use the new_node to absorb the other edge
                for ii in [0, 1]:

                    # if the downstream node has more than one incoming edges,
                    number_of_incoming_edges_in_downstream_node = G_copy.in_degree(downstreams[ii])
                    if number_of_incoming_edges_in_downstream_node > 1:

                        new_node = str(node) + "_dummy" + str(dummy_counter)
                        dummy_counter += 1

                        parent_cell_pointer[new_node] = str(node)

                        # the three lines below separate the merge and split
                        G_copy_2.remove_edge(node, downstreams[ii])
                        G_copy_2.add_edge(node, new_node)
                        G_copy_2.add_edge(new_node, downstreams[ii])

    return G_copy_2, dummy_counter, parent_cell_pointer


def apply_three_transforms(
    G,
    dummy_counter=1,
):
    """

    Args:
      G: input graph
      dummy_counter: as the name suggests (Default value = 1)

    Returns:
      G will be transformed using all the other two functions in this file
      and dummy_counter

    """

    G_copy = nx.DiGraph(G)
    parent_cell_pointer = {}

    G_copy, dummy_counter, parent_cell_pointer = split_diverging(
        G_copy,
        parent_cell_pointer,
        dummy_counter=dummy_counter,
    )
    G_copy, dummy_counter, parent_cell_pointer = split_merging(G_copy, parent_cell_pointer, dummy_counter=dummy_counter)

    G_copy, dummy_counter, parent_cell_pointer = remove_ME_DI_overlap(
        G_copy, parent_cell_pointer, dummy_counter=dummy_counter
    )

    return G_copy, dummy_counter, parent_cell_pointer
