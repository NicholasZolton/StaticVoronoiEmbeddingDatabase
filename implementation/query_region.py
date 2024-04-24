import pickle
from matplotlib import pyplot as plt
import numpy as np
from openTSNE import TSNE, TSNEEmbedding
from openai import OpenAI
from shapely.geometry import Polygon, Point
from shapely.strtree import STRtree


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

    # turn all the polygons into shapely polygons
    shapely_polygons = [Polygon(polygon) for polygon in polygons]

    # visualize the query point
    plt.scatter(new_point[0], new_point[1], c="red", s=5)

    # create the STRTree to query the polygons
    records = [
        {"geometry": polygon, "value": color}
        for polygon, color in zip(shapely_polygons, color_names)
    ]
    tree = STRtree([record["geometry"] for record in records])
    items = np.array([record["value"] for record in records])
    items.take(tree.query(Point(new_point))).tolist()
    print(
        "Nearest color: %s"
        % items.take(tree.query_nearest(Point(new_point[0], new_point[1]))).tolist()
    )

    plt.show()


if __name__ == "__main__":
    main()
