from typing import List

from matplotlib import pyplot as plt

from kirkpatrick_master.polygons import Point, Polygon


def plot_points(points: List[Point], *args, **kwargs):
    """
    Plot a list of points in a scatter plot style.

    :param points: list of Point objects

    :param args: var args to be passed to pyplot

    :param kwargs: kwargs to be passed to pyplot
    """
    xs = [p.x for p in points]
    ys = [p.y for p in points]

    plt.scatter(xs, ys, *args, **kwargs)


def plot_point(point: Point, *args, **kwargs):
    """
    Plot a single point.

    :param point: point to be plotted

    :param args: var args to be passed to pyplot

    :param kwargs: kwargs to be passed to pyplot
    """
    plot_points([point], *args, **kwargs)


def plot_polygon(polygon: Polygon, *args, **kwargs):
    """
    Plot a polygon.

    :param polygon: Polygon to be plotted

    :param args: var args to be passed to pyplot

    :param kwargs: kwargs to be passed to pyplot
    """
    if polygon is None:
        return

    points = polygon.pts
    cycle = points + [points[0]]

    xs = [p.x for p in cycle]
    ys = [p.y for p in cycle]

    plt.plot(xs, ys, *args, **kwargs)


def plot_polygons(polygons, *args, **kwargs):
    """
    Plot a list of polygons.

    :param polygons:

    :param args: var args to be passed to pyplot

    :param kwargs: kwargs to be passed to pyplot
    """
    for polygon in polygons:
        plot_polygon(polygon, *args, **kwargs)
