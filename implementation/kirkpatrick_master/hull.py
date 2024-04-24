from typing import List, Tuple

from kirkpatrick_master.polygons import Segment, Point
from kirkpatrick_master.utils import ccw
import logging
import numpy as np
from scipy.spatial import ConvexHull


def sq_dist2line(p: Point, line: Segment) -> float:
    """
    Compute squared distance from p to a line.

    :param p: Point

    :param line: Segment object defining a line

    :return: Squared dist from p to line.
    """
    A = line.A
    B = line.B
    a = A.y - B.y
    b = B.x - A.x
    c = A.x * (B.y - A.y + A.x - B.x)

    projection = a * p.x + b * p.y + c
    projection = projection * projection

    return projection / (a * a + b * b)


def farthest_point(A: Point, B: Point, points: List[Point]) -> Point:
    """
    Compute the point farthest from line defined by A, B

    :param A: one point defining line

    :param B: second point defining line

    :param points: list of candidate points

    :return: Point in points that is farthest from line
    """
    line = Segment(A, B)
    dists = [sq_dist2line(p, line) for p in points]

    p2dist = zip(points, dists)
    point, dist = max(((p, d) for p, d in p2dist), key=lambda x: x[1])

    return point


def split(A: Point, B: Point, points: List[Point]) -> Tuple[List[Point], List[Point]]:
    """
    Splits list of points into those above the line AB, and those below AB

    :param A: on point on line

    :param B: second point on line

    :param points: list of Points to split

    :return: List of points above line, list of points below line
    """
    points_up = [p for p in points if ccw(A, B, p) == 1]
    points_down = [p for p in points if ccw(A, B, p) == -1]

    return points_up, points_down


def split_tri(
    A: Point, u: Point, B: Point, points: List[Point]
) -> Tuple[List[Point], List[Point]]:
    """
    Splits a list of points around a triangle to find set of points above Au and above uB.

    :param A: first point on triangle

    :param u: second point on triangle

    :param B: third point on triangle

    :param points: list of points to split

    :return: points above Au, points above uB
    """
    above_Au, below_Au = split(A, u, points)
    above_uB, below_uB = split(u, B, points)

    return above_Au, above_uB


def upperhull(A: Point, B: Point, points: List[Point]) -> List[Point]:
    """
    Computes the upper hull of a list of points with A, B being the left and right bounding points on the set.

    :param A: left most point in points

    :param B: right most point in points

    :param points: list of points

    :return: list of points defining upper half of convex hull
    """
    if len(points) == 0:
        return []
    elif len(points) == 1:
        return [points[0]]
    elif len(points) == 2:
        return [points[0], points[1]]

    u = farthest_point(A, B, points)
    left_points, right_points = split_tri(A, u, B, points)
    logging.debug(f"Left points: {left_points}")
    logging.debug(f"Right points: {right_points}")

    return upperhull(u, B, right_points) + [u] + upperhull(A, u, left_points)


def quickhull(points: List[Point]) -> List[Point]:
    """
    Compute the convex hull of points using quickhull algorithm.

    :param points: list of points to use

    :return: subset of those points that define their convex hull.
    """
    coords = np.array([(p.x, p.y) for p in points])
    hull = ConvexHull(coords)
    hull_points = [Point(x, y) for x, y in coords[hull.vertices]]
    return hull_points
