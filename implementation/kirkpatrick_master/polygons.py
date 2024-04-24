from random import uniform, random, sample

import heapq
import math
import numpy as np
from scipy.spatial import Delaunay
from typing import List, Tuple
from kirkpatrick_master import utils


class Point:
    """A Point object that implements many basic vector operations."""

    def __init__(self, x: float, y: float):
        self.x = float(x)
        self.y = float(y)

    def __str__(self):
        return "Point(%f, %f)" % (self.x, self.y)

    def __repr__(self):
        return self.__str__()

    def __hash__(self):
        return hash((self.x, self.y))

    def __eq__(self, other):
        if isinstance(other, Point):
            return other.x == self.x and other.y == self.y
        else:
            raise ValueError()

    def __add__(self, other):
        if isinstance(other, Point):
            return Point(self.x + other.x, self.y + other.y)
        else:
            return Point(self.x + other, self.y + other)

    def __sub__(self, other):
        if isinstance(other, Point):
            return Point(self.x - other.x, self.y - other.y)
        else:
            return Point(self.x - other, self.y - other)

    def __mul__(self, other):
        if np.isscalar(other):
            return Point(self.x * other, self.y * other)
        raise ValueError("'other' must be of scalar type")

    def __rmul__(self, other):
        return self.__mul__(other)

    def __truediv__(self, other):
        return Point(self.x / other, self.y / other)

    def tuple(self):
        """Return a tuple (x, y) of the point"""
        return self.x, self.y

    @classmethod
    def sample_disk(cls, r):
        """Returns a point sampled from a disk of radius r"""
        r_squared = r**2
        sample_r_squared = uniform(0, r_squared)
        sample_theta = 2 * math.pi * random()
        x = math.sqrt(sample_r_squared) * math.cos(sample_theta)
        y = math.sqrt(sample_r_squared) * math.sin(sample_theta)
        return cls(x, y)

    @classmethod
    def sample_square(cls, a):
        """Returns a point sampled from a square of side length a"""
        x = uniform(0, a)
        y = uniform(0, a)
        return cls(x, y)


class Segment:
    def __init__(self, a: Point, b: Point):
        self.A = a
        self.B = b


class Polygon:
    """
    Encapsulates a general polygon and some manipulations.
    """

    def __init__(self, pts: List[Point]):
        """Initialize a polygon from a list of points in counter clockwise order"""
        assert len(pts) >= 3
        self.n = len(pts)
        self.pts = list(pts)
        self.segments = list(zip(self.pts, np.roll(self.pts, -1)))
        if not self.is_ccw():
            self.pts = self.pts[::-1]
            self.segments = list(zip(self.pts, np.roll(self.pts, -1)))
        self._x = np.array([p.x for p in pts])
        self._y = np.array([p.y for p in pts])
        self.np_pts = np.array(list(zip(self._x, self._y)))

        self.area = self._area()

        self.name = ""

    def __repr__(self):
        return "Polygon(" + repr(self.pts) + ")"

    def is_ear(self, ear: Tuple[int, int, int]) -> bool:
        """Return true if the 3 indices specify the indices of an ear in ccw order."""
        pts = np.array(self.pts)
        ear = list(ear)
        triples = zip(
            range(self.n), np.roll(range(self.n), -1), np.roll(range(self.n), -2)
        )
        reflex_vertices = {pts[t[1]] for t in triples if utils.ccw(*pts[list(t)]) == -1}
        if utils.ccw(*pts[ear]) != 1:
            return False
        if self.seg_intersect(pts[[ear[0], ear[2]]]):
            return False

        closure = Triangle(pts[ear])
        if any(closure.contains(v, closed=False) for v in reflex_vertices):
            return False

        return True

    def ears(self) -> List[int]:
        """Returns a list of the indices of the ears of the polygon"""
        pts = np.array(self.pts)
        triples = zip(
            range(self.n), np.roll(range(self.n), -1), np.roll(range(self.n), -2)
        )
        reflex_vertices = {
            pts[ear[1]] for ear in triples if utils.ccw(*pts[list(ear)]) == -1
        }

        ear_ixs = list()

        for i in range(self.n):
            ear = [(i - 1) % self.n, i, (i + 1) % self.n]
            if utils.ccw(*pts[ear]) != 1:
                continue
            if self.seg_intersect(pts[[ear[0], ear[2]]]):
                continue

            closure = Triangle(pts[ear])
            if any(closure.contains(v, closed=False) for v in reflex_vertices):
                continue

            ear_ixs.append(i)

        return ear_ixs

    def seg_intersect(self, segment: Tuple[Point, Point]) -> bool:
        """
        Return true if segment intersects the polygon, but does not share a vertex.
        """
        return any(
            utils.intersect(*segment, *edge, closed=False) for edge in self.segments
        )

    def is_ccw(self):
        """Return True if polygon is given in CCW order."""
        # Sum cross products of consecutive segments
        cross = sum((b.x - a.x) * (b.y + a.y) for a, b in self.segments)

        # If total cross product is negative, then CCW
        return cross < 0

    def _area(self) -> float:
        # https://en.wikipedia.org/wiki/Shoelace_formula
        return 0.5 * np.abs(
            np.dot(self._x + np.roll(self._x, 1), self._y - np.roll(self._y, 1))
        )

    def _rand_interior_point(self) -> Point:
        triangulation = Delaunay(self.np_pts)
        triangles = self.np_pts[triangulation.simplices]
        triangles = [Triangle([Point(*p) for p in tri]) for tri in triangles]
        total_area = sum(tri.area for tri in triangles)
        r = random()
        cumul = 0

        for tri in triangles:
            w = tri.area / total_area
            cumul += w
            if cumul > r:
                return tri._rand_interior_point()

    def _is_simple_split(self, u: int, p: Point, v: int) -> bool:
        """
        Return True if connecting u->p and p->v creates two mutually simple
        polygons within the original polygon.
        """
        u = self.pts[u]
        v = self.pts[v]

        # Checks if u->p or p->v intersects any side of the polygon.
        for i, j in self.segments:
            if u not in (i, j):
                if utils.intersect(u, p, i, j):
                    return False
            if v not in (i, j):
                if utils.intersect(v, p, i, j):
                    return False

        return True

    def _rand_split(self) -> Tuple[int, Point, int]:
        """
        Select 3 random points defining a split---no guarantees on simplicity of result.
        Return u, p, v where u, v are indices into self.pts, and p is a Point inside the polygon.
        We represent u, v by their indices in order to slice the boundary of the polygon more easily.
        """
        # Select two points on the polygon
        u, v = sample(range(self.n), 2)
        u, v = sorted((u, v))

        # Choose random interior point
        p = self._rand_interior_point()

        return u, p, v

    def split(self) -> Tuple["Polygon", "Polygon"]:
        """
        Split a polygon into two parts by choosing two random vertices,
        and connecting them to a random interior point.

        :return: two polygons
        """

        # TODO: Make this not crappy (if I can figure out)
        def _hlep() -> Tuple[List[Point], List[Point]]:
            u, p, v = self._rand_split()
            while not self._is_simple_split(u, p, v):
                u, p, v = self._rand_split()
            pts1 = self.pts[u : v + 1] + [p]
            pts2 = self.pts[v:] + self.pts[: u + 1] + [p]
            return pts1, pts2

        # I thought that the above method would give me two random polygons,
        # But it can mess up and the polygons are overlapping.  I get around this
        # with a hacky method of simply re-trying until it works.
        pts1, pts2 = _hlep()
        poly1 = Polygon(pts1)
        poly2 = Polygon(pts2)
        while poly1.area + poly2.area > self.area:
            pts1, pts2 = _hlep()
            poly1 = Polygon(pts1)
            poly2 = Polygon(pts2)

        return poly1, poly2

    def bounding_rect(self) -> Tuple[int, int, int, int]:
        """Return the 4-tuple of bounding values: min x, max x, min y, max y"""
        min_x = np.min(self._x)
        max_x = np.max(self._x)
        min_y = np.min(self._y)
        max_y = np.max(self._y)

        return min_x, max_x, min_y, max_y


class Triangle(Polygon):
    def __init__(self, pts):
        super().__init__(pts)
        assert self.n == 3

    def __repr__(self):
        return "Triangle(" + repr(self.pts) + ")"

    def contains(self, item, closed=True):
        if isinstance(item, Point):
            point = item
            ccw_01p = utils.ccw(self.pts[0], self.pts[1], point)
            ccw_12p = utils.ccw(self.pts[1], self.pts[2], point)
            ccw_20p = utils.ccw(self.pts[2], self.pts[0], point)
            if closed:
                if point in self.pts:
                    return True
                if ccw_01p * ccw_12p * ccw_20p == 0:
                    # If any is 0, then p is on the edge of triangle
                    return True
            return ccw_01p == ccw_12p == ccw_20p
        else:
            raise ValueError("item must be instance of [Point]")

    def __contains__(self, item):
        return self.contains(item)

    def scale(self, scale=1) -> "Triangle":
        center = sum(self.pts, Point(0, 0)) / 3
        deltas = [vec - center for vec in self.pts]
        scaled = [vec * scale for vec in deltas]
        new_pts = [vec + center for vec in scaled]
        return Triangle(new_pts)

    def _rand_interior_point(self):
        v0 = self.pts[0]
        v1 = self.pts[1] - self.pts[0]
        v2 = self.pts[2] - self.pts[0]
        while True:
            r1 = random()
            r2 = random()
            p = v0 + r1 * v1 + r2 * v2
            if p in self:
                return p

    @classmethod
    def enclosing_triangle(cls, poly):
        min_x, max_x, min_y, max_y = poly.bounding_rect()
        rect_w = max_x - min_x
        rect_h = max_y - min_y

        # Basic geometry tells us that a bounding equilateral triangle on the rectangle is
        # tri_side_len = rect_w + rect_h * 2 / math.sqrt(3)
        delta = rect_h / math.sqrt(3)  # Gap between bottom corners of rect and triangle

        a = Point(min_x, min_y) - Point(delta, 0)
        b = Point(max_x, min_y) + Point(delta, 0)
        c = (Point(min_x, min_y) + Point(max_x, min_y)) / 2 + Point(
            0, math.sqrt(3) * rect_w / 2 + rect_h
        )

        return cls([a, b, c])


def generate_random_tiling(polygon: Polygon, n_iter: int = 10) -> List[Polygon]:
    """
    Generates a tiling by iteratively splitting the largest tile, starting with a single polygon.
    This method is quite slow and does not always seem to terminate due to the difficulty in finding a simple
    splitting point.
    """
    heap = [(-1 * polygon.area, polygon)]
    for i in range(n_iter):
        neg_area, poly = heapq.heappop(heap)
        tile1, tile2 = poly.split()
        heapq.heappush(heap, (-1 * tile1.area, tile1))
        heapq.heappush(heap, (-1 * tile2.area, tile2))

    return [poly for neg_area, poly in heap]


def generate_triangle_tiling(num_pts: int = 100, size: int = 100) -> List[Triangle]:
    """
    Generate a larger triangular tiling by creating many points, and finding the Delaunay triangulation.

    :param num_pts: the number of points in the tiling

    :param size: the size square from which to sample points
    """
    pts = [Point.sample_square(size) for i in range(num_pts)]
    np_pts = np.array([p.tuple() for p in pts])
    pts = np.array(pts)
    triangulation = Delaunay(np_pts)
    triangles = pts[triangulation.simplices]
    triangles = [Triangle(triple.tolist()) for triple in triangles]
    return triangles
    # side = total_side_len / squares_per_side
    # tiles = [
    #     Polygon([Point(x, y), Point(x + side, y), Point(x + side, y + side), Point(x, y + side)])
    #     for x, y in product(np.arange(0, total_side_len, side), repeat=2)
    # ]
    # return tiles


if __name__ == "__main__":
    tris = generate_triangle_tiling(100, 100)
    print(tris)
