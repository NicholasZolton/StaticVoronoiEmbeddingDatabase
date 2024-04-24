from typing import Set, List

from networkx import Graph

from kirkpatrick_master.polygons import Point

INDEPENDENT_SET_LEMMA_DEGREE = 8


def planar_independent_set(
    graph: Graph, black_list: List = list(), degree_lim=INDEPENDENT_SET_LEMMA_DEGREE
) -> Set[Point]:
    """
    Assuming the graph is planar, then computes an independent set of size at
    least n/18 in which every node has degree at most 8.  O(n) time.

    (Note: assume the graph is planar because it is computationally expensive to check)

    :param graph: networkx Graph (assumed planar)

    :param black_list: list of nodes that should not be in the independent set

    :param degree_lim: nodes in independent set have degree at most this

    :return: set of graph nodes
    """
    unmarked_nodes = {
        node
        for node in graph.nodes_iter()
        if graph.degree(node) <= degree_lim and node not in black_list
    }

    independent_set = set()

    while len(unmarked_nodes) > 0:
        node = unmarked_nodes.pop()
        independent_set.add(node)
        unmarked_nodes -= set(graph.neighbors(node))

    return independent_set
