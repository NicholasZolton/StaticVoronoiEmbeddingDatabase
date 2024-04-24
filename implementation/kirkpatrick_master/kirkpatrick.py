from collections import defaultdict

import logging
from matplotlib import pyplot as plt
import numpy as np
import time
from networkx import DiGraph, Graph
import networkx as nx
from typing import List, Tuple, Optional

from kirkpatrick_master import drawing
from kirkpatrick_master.hull import quickhull
from kirkpatrick_master.independentset import planar_independent_set
from kirkpatrick_master.polygons import Point, Polygon, Triangle
from kirkpatrick_master.polygons import generate_random_tiling, generate_triangle_tiling
from kirkpatrick_master.triangulate import triangulate


def wrap_triangle(poly: Polygon) -> Tuple[Triangle, List[Triangle]]:
    """
    Finds a large triangle that surrounds the polygon, and tiles the gap
    between the triangle the polygon.

    :param poly: polygon to be surrounded

    :return: tuple of the bounding triangle and the triangles in the gap
    """
    """Wraps the polygon in a triangle and triangulates the gap"""
    bounding_triangle = Triangle.enclosing_triangle(poly)

    # We need to expand the triangle a little bit so that the polygon side isn't on top of
    # one of the triangle sides (otherwise Triangle library segfaults).
    bounding_triangle = bounding_triangle.scale(1.1)

    # bounding_region = bounding_triangle.triangulate(hole=poly)
    bounding_region = triangulate(bounding_triangle, hole=poly)
    return bounding_triangle, bounding_region


def triangle_graph(regions: List[Polygon], graph: Graph) -> List[Triangle]:
    """
    Triangulate regions, connecting into a tree (acyclic digraph) for lookups from triangles to
    the original polygon tiling.

    :param regions:  list of regions to be triangulated

    :param graph: graph to be populated with edge from the original polygons to the triangles in their triangulations.

    :return: list of all triangles in all regions
    """
    logging.debug("Triangulating subdivision of %d regions" % len(regions))
    triangles = []
    for i, region in enumerate(regions):
        logging.debug("Triangulating region %d" % i)
        logging.getLogger().handlers[0].flush()
        graph.add_node(region, original=True)

        if isinstance(region, Triangle):
            triangles.append(region)
        elif region.n == 3:
            region.__class__ = Triangle
            triangles.append(region)
        else:
            # triangulation = region.triangulate()
            triangulation = triangulate(region)
            for triangle in triangulation:
                graph.add_node(triangle, original=False)
                graph.add_edge(triangle, region)
                triangles.append(triangle)

    return triangles


def remove_point_triangulation(affected_triangles: List[Triangle], p: Point) -> Polygon:
    """
    Removes a point from affected triangles, return the resulting polygon that fills the gap.

    :param affected_triangles: list of the triangles containing the point to be removed

    :param p: point to be removed

    :return: polygon created by merging the affected triangles
    """

    logging.debug("Removing point from triangulation")

    # First we construct a dictionary that tells us adjacency of triangles
    boundaries = [set(tri.pts) for tri in affected_triangles]
    point2triangles = defaultdict(set)
    for i, bound in enumerate(boundaries):
        bound.remove(p)
        u, v = bound
        point2triangles[u].add(i)
        point2triangles[v].add(i)

    # Connect adjacent triangles, noting which point connects them
    graph = Graph()
    try:
        for u, (i, j) in point2triangles.items():
            graph.add_edge(i, j, point=u)
    except Exception as e:
        raise e

    # Walk around the triangles to get the new outer boundary
    # TODO: Remember to make this work.  DFS visits all nodes not all edges.  I think find_cycle works.
    #
    new_boundary = [
        graph.get_edge_data(i, j)["point"]
        for (i, j) in nx.find_cycle(graph)
        # for (i, j) in nx.dfs_edges(graph)
    ]

    return Polygon(new_boundary)


def next_layer(
    regions: List[Triangle], boundary: Triangle, digraph: DiGraph
) -> List[Triangle]:
    """
    Compute the next layer in the data structure by removing O(n) points, retriangulating,
    connecting new triangles in DAG to triangles with which they might overlap.  We don't
    compute actual intersections to save time.

    :param regions: the current layer in the algorithm

    :param boundary: the bounding triangle (so that the boundary is never removed)

    :param digraph: digraph search tree for later location

    :return: list of triangles in the next layer
    """
    # Since a tiling is represented as list of polygons, points may
    # appear multiple times in the tiling.  We produce a mapping to
    # bring all information together
    point2regions = defaultdict(set)
    for i, region in enumerate(regions):
        for point in region.pts:
            point2regions[point].add(i)

    # Graph on the vertices of the tiling
    graph = Graph()
    for region in regions:
        for u, v in zip(region.pts, np.roll(region.pts, 1)):
            graph.add_edge(u, v)

    # Find independent set to remove constant fraction of triangles
    ind_set = planar_independent_set(graph, black_list=boundary.pts)

    # Find the affected regions to be joined together, triangulate, and connect into DAG
    unaffected = set(range(len(regions)))
    new_regions = list()
    for point in ind_set:
        # Remove point and join triangles.
        affected_ixs = point2regions[point]
        unaffected.difference_update(affected_ixs)
        affected = [regions[i] for i in affected_ixs]
        new_poly = remove_point_triangulation(affected, point)

        # Retriangulate
        # new_triangles = new_poly.triangulate()
        new_triangles = triangulate(new_poly)
        new_regions += new_triangles

        # Connect into DAG for lookups
        for tri in new_triangles:
            for ix in affected_ixs:
                digraph.add_node(tri, original=False)
                digraph.add_edge(tri, regions[ix])

    new_regions += [regions[i] for i in unaffected]
    return new_regions


class Kirkpatrick:
    """
    Implementation of Kirkpatrick's algorithm.  When passed a tiling of
    polygons, it processes it to produce a search tree.  After preprocessing,
    location takes O(log n) time, and the object uses O(n) space.

    The plot_layers parameter will produce png files of the layers the
    algorithm produces during preprocessing.  Note: this takes a while.
    """

    def __init__(self, subdivision: List[Polygon], plot_layers=False):
        """
        Create a point locator object on a planar subdivision
        """
        self.digraph = DiGraph()
        self.top_layer = list()
        self._preprocess(subdivision, plot_layers=plot_layers)

    def _preprocess(self, subdivision: List[Polygon], plot_layers=False):
        """
        If subdivision is not triangular, then triangulate each non-triangle region.
        Then place large triangle around region, and triangulate
        :param subdivision:
        """
        logging.debug("Preprocessing planar subdivision")
        subdivision = triangle_graph(subdivision, self.digraph)

        logging.debug("Received triangulated subdivision")

        logging.debug("Constructing convex hull")
        all_pts = {p for tri in subdivision for p in tri.pts}

        logging.debug("Got Points")
        hull = quickhull(list(all_pts))
        logging.debug("Got Hull")
        hull = Polygon(hull)
        logging.debug("Got Polygon")

        logging.debug("Wrapping polygon in bounding triangle")
        bounding_tri, gap_triangles = wrap_triangle(hull)

        for tri in gap_triangles:
            self.digraph.add_node(tri, original=False)

        layer = subdivision + gap_triangles
        if plot_layers:
            drawing.plot_polygons(layer, "k-")
            plt.savefig("layer0.png")
            plt.clf()

        logging.debug("Iterating over layers")
        i = 0
        while len(layer) > 1:
            logging.debug("Current layer size: %d" % len(layer))
            layer = next_layer(layer, bounding_tri, self.digraph)
            i += 1
            if plot_layers:
                drawing.plot_polygons(layer, "k-")
                plt.savefig("layer%d.png" % i)
                plt.clf()

        logging.debug("Final layer size: %d" % len(layer))
        self.top_layer = layer

    def locate(self, p: Point, plot_search=False) -> Optional[Polygon]:
        """
        Locates a point in the original tiling in O(log n) time.

        :param p: point to be located

        :param plot_search: plots the tiles it searchs on the way

        :return: either the original polygon, or None if outside the tiling
        """
        curr = None
        for region in self.top_layer:
            if p in region:
                curr = region
                break
        else:
            return None

        # Iterate until the layer of triangles immediately above the original tiling
        # This is because it is easy to test point containment in a triangle, not easy in general polygons.
        if plot_search:
            drawing.plot_point(p)
            drawing.plot_polygon(curr, "r-")
            plt.savefig("search_layer0.png")
            plt.clf()

        i = 1
        while len(self.digraph.neighbors(curr)) > 1:
            for node in self.digraph.neighbors(curr):
                if p in node:
                    curr = node
                    break
            else:
                return None
            if plot_search:
                drawing.plot_point(p)
                highlights = list()
                for node in self.digraph.neighbors(curr):
                    drawing.plot_polygon(node, "k-")
                    if node.n == 3 and p in node:
                        highlights.append(node)
                drawing.plot_polygons(highlights, "r-")
                plt.savefig("search_layer%d.png" % i)
                plt.clf()
            i += 1

        # Access the original tile just below the polygon if it exists
        neighbors = self.digraph.neighbors(curr)
        if len(neighbors) == 1:
            curr = neighbors[0]

        if self.digraph.node[curr]["original"]:
            return curr
        return None


def time_tests(min_pts: int = 10, max_pts: int = 100, inc=5, n_iter=100) -> List[Point]:
    """
    Executes an intensive test of the algorithm.  Generates many tilings of
    different numbers of points, and generates many query points to test.

    :param min_pts: mininum number of points in tiling

    :param max_pts: maximum number of points in tiling

    :param inc: generate tilings every inc points from min to max

    :param n_iter: number of query points to test

    :return: List of points of (num_tiles, time for a query)
    """
    logging.info("Running timing tests on point location")
    size = 100000
    data = list()
    for i in range(min_pts, max_pts, inc):
        logging.info("Performing tests on %d points" % i)
        tiles = generate_triangle_tiling(num_pts=i, size=size)
        locator = Kirkpatrick(tiles)
        for j in range(n_iter):
            query = Point.sample_square(size)
            start = time.time()
            locator.locate(query)
            elapsed = time.time() - start
            data.append(Point(len(tiles), elapsed))
    return data


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    tiles = [Polygon([Point(0, 0), Point(2, 0), Point(2, 2), Point(0, 2)])]

    locator = Kirkpatrick(tiles)

    query_point = Point(1, 1)

    located_tile = locator.locate(query_point, plot_search=True)

    print(located_tile)
