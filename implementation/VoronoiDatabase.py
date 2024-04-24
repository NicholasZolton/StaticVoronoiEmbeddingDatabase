import pickle
from matplotlib import pyplot as plt
from openTSNE import TSNE, TSNEEmbedding
from scipy.spatial import Voronoi, voronoi_plot_2d
import openTSNE
from openai import OpenAI
import numpy as np
from shapely.geometry import Polygon, Point
from shapely.strtree import STRtree


def voronoi_finite_polygons_2d(vor, plt, points, radius=None):
    """
    Reconstruct infinite voronoi regions in a 2D diagram to finite
    regions.

    Parameters
    ----------
    vor : Voronoi
        Input diagram
    radius : float, optional
        Distance to 'points at infinity'.

    Returns
    -------
    regions : list of tuples
        Indices of vertices in each revised Voronoi regions.
    vertices : list of tuples
        Coordinates for revised Voronoi vertices. Same as coordinates
        of input vertices, with 'points at infinity' appended to the
        end.

    """

    if vor.points.shape[1] != 2:
        raise ValueError("Requires 2D input")

    new_regions = []
    new_vertices = vor.vertices.tolist()

    center = vor.points.mean(axis=0)
    if radius is None:
        radius = vor.points.ptp().max()

    # Construct a map containing all ridges for a given point
    all_ridges = {}
    for (p1, p2), (v1, v2) in zip(vor.ridge_points, vor.ridge_vertices):
        all_ridges.setdefault(p1, []).append((p2, v1, v2))
        all_ridges.setdefault(p2, []).append((p1, v1, v2))

    # Reconstruct infinite regions
    for p1, region in enumerate(vor.point_region):
        vertices = vor.regions[region]

        if all(v >= 0 for v in vertices):
            # finite region
            new_regions.append(vertices)
            continue

        # reconstruct a non-finite region
        ridges = all_ridges[p1]
        new_region = [v for v in vertices if v >= 0]

        for p2, v1, v2 in ridges:
            if v2 < 0:
                v1, v2 = v2, v1
            if v1 >= 0:
                # finite ridge: already in the region
                continue

            # Compute the missing endpoint of an infinite ridge

            t = vor.points[p2] - vor.points[p1]  # tangent
            t /= np.linalg.norm(t)
            n = np.array([-t[1], t[0]])  # normal

            midpoint = vor.points[[p1, p2]].mean(axis=0)
            direction = np.sign(np.dot(midpoint - center, n)) * n
            far_point = vor.vertices[v2] + direction * radius

            new_region.append(len(new_vertices))
            new_vertices.append(far_point.tolist())

        # sort region counterclockwise
        vs = np.asarray([new_vertices[v] for v in new_region])
        c = vs.mean(axis=0)
        angles = np.arctan2(vs[:, 1] - c[1], vs[:, 0] - c[0])
        new_region = np.array(new_region)[np.argsort(angles)]

        # finish
        new_regions.append(new_region.tolist())

    regions = new_regions
    vertices = np.asarray(new_vertices)

    return regions, vertices


class NDVoronoiDatabase:
    """
    This class is responsible for storing the voronoi regions, vertices,
    coordinates, TSNE Embedding, and Voronoi object.
    """

    def __init__(self, initial_data: list[str] = None, colors: list[str] = None):
        self.regions: list[list[int]] = []
        self.vertices: list[list[float]] = []
        self.tsne_model: TSNEEmbedding | None = None
        self.coordinates: list[tuple[float, float]] = []
        self.coordinates_with_colors: list[tuple[float, float, str]] = []
        self.voronoi: Voronoi | None = None
        self.colors = colors
        self.polygons = []
        self.initial_data = initial_data

        # first step is to create the embeddings
        client = OpenAI()
        embedded_inputs = client.embeddings.create(
            model="text-embedding-3-small", input=initial_data
        )
        embeddings = [embedding.embedding for embedding in embedded_inputs.data]
        high_dimensional_embeddings = np.array(embeddings)
        affinites = openTSNE.affinity.PerplexityBasedNN(high_dimensional_embeddings)
        init = openTSNE.initialization.pca(high_dimensional_embeddings)
        self.tsne_model: TSNEEmbedding = openTSNE.TSNE().fit(
            affinities=affinites, initialization=init
        )

        points = []
        lower_dimensional_embeddings = self.tsne_model
        if colors is not None:
            for i in range(lower_dimensional_embeddings.__len__()):
                points.append(
                    (
                        lower_dimensional_embeddings[i][0],
                        lower_dimensional_embeddings[i][1],
                        colors[i].strip(),
                    )
                )
        else:
            for i in range(lower_dimensional_embeddings.__len__()):
                points.append(
                    (
                        lower_dimensional_embeddings[i][0],
                        lower_dimensional_embeddings[i][1],
                        # random color
                        "#" + "%06x" % np.random.randint(0, 0xFFFFFF),
                    )
                )

        self.coordinates_with_colors = points
        coordinates = np.array([(point[0], point[1]) for point in points])
        self.voronoi = Voronoi(coordinates)

        self.regions, self.vertices = voronoi_finite_polygons_2d(
            self.voronoi, plt, points=coordinates
        )

        self.polygons = []
        for index, region in enumerate(self.regions):
            polygon = self.vertices[region]
            self.polygons.append(polygon)
            plt.fill(
                *zip(*polygon), alpha=0.4, c=self.coordinates_with_colors[index][2]
            )
        self.shapely_polygons = [Polygon(polygon) for polygon in self.polygons]
        self.records = [
            {"geometry": polygon, "value": name}
            for polygon, name in zip(self.shapely_polygons, self.initial_data)
        ]
        self.str_tree = STRtree([record["geometry"] for record in self.records])
        self.items = np.array([record["value"] for record in self.records])

    def visualize_coordinates(self, plot):
        """
        Visualize the coordinates of the Voronoi diagram.
        This does not show the Voronoi diagram itself.
        """

        for point in self.coordinates_with_colors:
            try:
                plot.scatter(point[0], point[1], color=point[2])
            except:
                print("Invalid point: " + point)

        return plot

    def visualize_voronoi(self):
        """
        Visualize the Voronoi diagram.
        """

        for index, polygon in enumerate(self.polygons):
            plt.fill(
                *zip(*polygon), alpha=0.4, c=self.coordinates_with_colors[index][2]
            )

        for point in self.coordinates_with_colors:
            plt.scatter(point[0], point[1], color=point[2], s=0.8)

        plt.xlim(self.voronoi.min_bound[0] - 0.1, self.voronoi.max_bound[0] + 0.1)
        plt.ylim(self.voronoi.min_bound[1] - 0.1, self.voronoi.max_bound[1] + 0.1)

    # TODO: make this method return the site of the query
    def query_input(self, input_string: str, visualize: bool = False):
        """
        Query the voronoi diagram with a string.
        """
        client = OpenAI()
        query_embedding = (
            client.embeddings.create(model="text-embedding-3-small", input=input_string)
            .data[0]
            .embedding
        )
        transformed_point = self.tsne_model.transform([query_embedding])[0]

        if visualize:
            self.visualize_voronoi()
            plt.scatter(transformed_point[0], transformed_point[1], c="red", s=10)
            plt.show()

        return self.items.take(
            self.str_tree.query(Point(transformed_point[0], transformed_point[1]))
        ).tolist()

    def visualize_query(self, query_point: list[float], plot):
        """
        Visualize a query point on the Voronoi diagram.
        """
        plot.scatter(query_point[0], query_point[1], c="red", s=10)
        return plot

    def save(self, filename: str):
        """
        Save the Voronoi database to a file.
        """
        with open(filename, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(filename: str) -> "NDVoronoiDatabase":
        """
        Load a Voronoi database from a file.
        """
        with open(filename, "rb") as f:
            return pickle.load(f)
