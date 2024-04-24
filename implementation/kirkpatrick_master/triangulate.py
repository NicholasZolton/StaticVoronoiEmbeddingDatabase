from itertools import product

from matplotlib import pyplot as plt

import numpy as np
from typing import List

from kirkpatrick_master import drawing
from kirkpatrick_master import utils
from kirkpatrick_master.polygons import Point, Polygon, Triangle


def _seg_intersect_poly(seg, poly):
    return any(utils.intersect(*seg, *edge, closed=False) for edge in poly.segments)


def _remove_hole(poly: Polygon, hole: Polygon) -> Polygon:
    """
    Removes hole from polygon by turning it into a single degenerate polygon
    with a bridge edge connecting poly to interior hole
    :param poly:
    :param hole:
    :return:
    """
    possible_bridges = list(product(poly.pts, hole.pts))
    possible_bridges.sort(key=lambda b: utils.dist(*b))
    bridge = None
    for bridge in possible_bridges:
        if _seg_intersect_poly(bridge, poly):
            continue
        if _seg_intersect_poly(bridge, hole):
            continue

        break

    poly_pt, hole_pt = bridge

    # Roll back the poly points to go around polygon first, away from bridge
    poly_ix = poly.pts.index(poly_pt)
    poly_pts = list(np.roll(poly.pts, -poly_ix))

    # Reverse and roll the hole points because we must go around hole c.w. to
    # go around whole degenerate polygon ccw
    hole_pts = hole.pts[::-1]
    hole_ix = hole_pts.index(hole_pt)
    hole_pts = list(np.roll(hole_pts, -hole_ix))

    degenerate_pts = poly_pts + [poly_pt] + hole_pts + [hole_pt]
    return Polygon(degenerate_pts)


def _is_valid_ear(ear: List[int], poly):
    pts = np.array(poly.pts)
    return utils.ccw(*pts[ear]) == 1 and not _seg_intersect_poly(
        pts[[ear[0], ear[2]]], poly
    )


def triangulate(poly: Polygon, hole: Polygon = None) -> List[Triangle]:
    """
    Triangulates a polygon, potentially with a hole in it.  Uses naive O(n^2)
    ear-clipping method.

    :param poly: polygon to be triangulated

    :param hole: hole in the polygon

    :return: list of triangles making up the polygon
    """
    if hole:
        poly = _remove_hole(poly, hole)

    n = poly.n
    curr_n = n
    pts = np.array(poly.pts)

    ears = {ear: [(ear - 1) % n, ear, (ear + 1) % n] for ear in poly.ears()}

    # Adjacency dict of points in poly
    adj = {i: ((i - 1) % n, (i + 1) % n) for i in range(n)}

    tris = list()
    while len(tris) < n - 2:
        # Pick a random ear, turn into triangle, and append it to triangulation
        b, ear = ears.popitem()
        tris.append(Triangle(pts[ear].tolist()))

        # Update connection of vertices adjacent to ear vertex
        a, b, c = ear
        adj[a] = (adj[a][0], c)
        adj[c] = (a, adj[c][1])

        # Update ear status of adjacent vertices
        ear_a = (adj[a][0], a, c)
        ear_c = (a, c, adj[c][1])
        if poly.is_ear(ear_a):
            ears[a] = list(ear_a)
        else:
            ears.pop(a, None)

        if poly.is_ear(ear_c):
            ears[c] = list(ear_c)
        else:
            ears.pop(c, None)

        curr_n -= 1

    return tris


if __name__ == "__main__":
    poly = Polygon(
        [
            Point(0.000000, 0.000000),
            Point(1.727261, 0.681506),
            Point(2.000000, 2.000000),
            Point(1.128893, 1.104487),
            Point(0.848083, 1.122645),
        ]
    )

    triangles = triangulate(poly)

    drawing.plot_polygons(triangles)
    plt.show()
