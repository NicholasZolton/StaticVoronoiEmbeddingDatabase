import pickle
from matplotlib import pyplot as plt
from openTSNE import TSNE, TSNEEmbedding
from scipy.spatial import Voronoi, voronoi_plot_2d
import openTSNE
from openai import OpenAI
import numpy as np


class AIVoronoiDatabase:
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
        for i in range(lower_dimensional_embeddings.__len__()):
            points.append(
                (
                    lower_dimensional_embeddings[i][0],
                    lower_dimensional_embeddings[i][1],
                    colors[i].strip(),
                )
            )

        self.coordinates_with_colors = points
        coordinates = np.array([(point[0], point[1]) for point in points])
        self.voronoi = Voronoi(coordinates)

        self.vertices = self.voronoi.vertices.tolist()

        center = self.voronoi.points.mean(axis=0)
        radius = self.voronoi.points.ptp().max()

        # create map containing all ridges for a given point
        all_ridges = {}
        for (p1, p2), (v1, v2) in zip(
            self.voronoi.ridge_points, self.voronoi.ridge_vertices
        ):
            all_ridges.setdefault(p1, []).append((p2, v1, v2))
            all_ridges.setdefault(p2, []).append((p1, v1, v2))

        for p1, region in enumerate(self.voronoi.point_region):
            vertices = self.voronoi.regions[region]

            if all(v >= 0 for v in vertices):
                self.regions.append(vertices)
                continue

            ridges = all_ridges[p1]
            new_region = [v for v in vertices if v >= 0]

            for p2, v1, v2 in ridges:
                if v2 < 0:
                    v1, v2 = v2, v1
                if v1 >= 0:
                    continue

                t = self.voronoi.points[p2] - self.voronoi.points[p1]
                t /= np.linalg.norm(t)
                n = np.array([-t[1], t[0]])

                midpoint = self.voronoi.points[[p1, p2]].mean(axis=0)
                direction = np.sign(np.dot(midpoint - center, n)) * n
                far_point = self.voronoi.vertices[v2] + direction * radius

                new_region.append(len(self.vertices))
                self.vertices.append(far_point.tolist())

            vs = np.asarray([self.vertices[v] for v in new_region])
            c = vs.mean(axis=0)
            angles = np.arctan2(vs[:, 1] - c[1], vs[:, 0] - c[0])
            new_region = np.array(new_region)[np.argsort(angles)]
            self.regions.append(new_region)

        self.polygons = [self.vertices[region] for region in self.regions]

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

    def visualize_voronoi(self, plot):
        """
        Visualize the Voronoi diagram.
        """

        for index, polygon in enumerate(self.polygons):
            plot.fill(
                *zip(*polygon), alpha=0.4, c=self.coordinates_with_colors[index][2]
            )

        plot.scatter(
            self.coordinates[:, 0],
            self.coordinates[:, 1],
            s=0.8,
            c=self.coordinates_with_colors[:, 2],
        )
        plot.xlim(self.voronoi.min_bound[0] - 0.1, self.voronoi.max_bound[0] + 0.1)
        plot.ylim(self.voronoi.min_bound[1] - 0.1, self.voronoi.max_bound[1] + 0.1)
        return plot

    # TODO: make this method return the site of the query
    def query_input(self, input_string: str, color: str):
        """
        Query the voronoi diagram with a string.
        """
        client = OpenAI()
        query_embedding = (
            client.embeddings.create(model="text-embedding-3-small", input=input_string)
            .data[0]
            .embedding
        )

        transformed_point = self.tsne_model.transform([query_embedding])

        return transformed_point[0]

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
    def load(filename: str) -> "AIVoronoiDatabase":
        """
        Load a Voronoi database from a file.
        """
        with open(filename, "rb") as f:
            return pickle.load(f)
