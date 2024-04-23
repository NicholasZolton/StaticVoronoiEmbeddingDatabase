import pickle
from matplotlib import pyplot as plt
from openTSNE import TSNE, TSNEEmbedding
from openai import OpenAI


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
    lime = "color: rainbow"

    client = OpenAI()

    embedded_color = (
        client.embeddings.create(model="text-embedding-3-small", input=lime)
        .data[0]
        .embedding
    )

    input_file = open("colors.csv", "r")
    all_lines = input_file.readlines()
    colors = []
    for line in all_lines:
        split_line = line.split(",")
        colors.append(split_line[1].strip())
    input_file.close()

    # colorize
    for index, region in enumerate(regions):
        polygon = vertices[region]
        plt.fill(*zip(*polygon), alpha=0.4, c=colors[index])

    plt.scatter(coordinates[:, 0], coordinates[:, 1], c=colors, s=0.8)
    plt.xlim(voronoi.min_bound[0] - 0.1, voronoi.max_bound[0] + 0.1)
    plt.ylim(voronoi.min_bound[1] - 0.1, voronoi.max_bound[1] + 0.1)

    # now get the new point using the TSNE model
    new_point = tsne_model.transform([embedded_color])
    print(new_point)

    # plot the new point
    plt.scatter(new_point[:, 0], new_point[:, 1], c="red", s=10)

    plt.show()


if __name__ == "__main__":
    main()
