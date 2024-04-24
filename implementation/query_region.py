import pickle
from matplotlib import pyplot as plt
from openTSNE import TSNE, TSNEEmbedding
from openai import OpenAI
from kirkpatrick_master.kirkpatrick import Kirkpatrick
from kirkpatrick_master.polygons import Point, Polygon
from kirkpatrick_master.drawing import plot_polygon, plot_polygons
from kirkpatrick_master.hull import quickhull
import logging

logging.basicConfig(level=logging.DEBUG)


def main():
    regions = None
    vertices = None
    tsne_model = None
    coordinates = None
    voronoi = None
    with open("regions.pkl", "rb") as f:
        regions = pickle.load(f)
    with open("vertices.pkl", "rb") as f:
        vertices = pickle.load(f)
    with open("tsne_model.pkl", "rb") as f:
        tsne_model: TSNEEmbedding = pickle.load(f)
    with open("coordinates.pkl", "rb") as f:
        coordinates = pickle.load(f)
    with open("voronoi.pkl", "rb") as f:
        voronoi = pickle.load(f)

    # try to add a new point to the tsne model
    lime = "color: lime"

    client = OpenAI()

    embedded_color = (
        client.embeddings.create(model="text-embedding-3-small", input=lime)
        .data[0]
        .embedding
    )

    input_file = open("colors.csv", "r")
    all_lines = input_file.readlines()
    colors = []
    color_names = []
    for line in all_lines:
        split_line = line.split(",")
        colors.append(split_line[1].strip())
        color_names.append(split_line[0].strip())
    input_file.close()

    polygons = []
    for index, region in enumerate(regions):
        polygon = vertices[region]
        polygons.append(polygon)
        plt.fill(*zip(*polygon), alpha=0.4, c=colors[index])

    plt.scatter(coordinates[:, 0], coordinates[:, 1], c=colors, s=0.8)
    plt.xlim(voronoi.min_bound[0] - 2.0, voronoi.max_bound[0] + 2.0)
    plt.ylim(voronoi.min_bound[1] - 2.0, voronoi.max_bound[1] + 2.0)

    # now get the new point using the TSNE model
    new_point = tsne_model.transform([embedded_color])[0]
    print(new_point)

    # plot the new point
    plt.scatter(new_point[0], new_point[1], c="red", s=10)

    # turn the polygons into kirkpatrick polygons
    kirkpatrick_polygons = []
    for index, polygon in enumerate(polygons):
        polygon_points = []
        for point in polygon:
            polygon_points.append(Point(point[0], point[1]))

        # make sure the polygon is closed by performing quick hull
        hull = quickhull(polygon_points)
        polygon_points = hull
        kirkpatrick_polygon = Polygon(polygon_points)

        kirkpatrick_polygon.name = color_names[index]
        kirkpatrick_polygons.append(kirkpatrick_polygon)

    # add a bounding box
    kirkpatrick_polygons.append(
        Polygon(
            [
                Point(voronoi.min_bound[0] - 1.0, voronoi.min_bound[1] - 1.0),
                Point(voronoi.max_bound[0] + 1.0, voronoi.min_bound[1] - 1.0),
                Point(voronoi.max_bound[0] + 1.0, voronoi.max_bound[1] + 1.0),
                Point(voronoi.min_bound[0] - 1.0, voronoi.max_bound[1] + 1.0),
            ]
        )
    )

    # plot_polygons(kirkpatrick_polygons)

    # create the kirkpatrick data structure
    locator = Kirkpatrick(kirkpatrick_polygons)
    print("data structure made")
    # query_point = Point(new_point[0], new_point[1])
    # located_tile = locator.locate(query_point)
    # print(located_tile)
    # print("point located")
    plt.show()


if __name__ == "__main__":
    main()
