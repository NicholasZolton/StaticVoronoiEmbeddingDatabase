# from sklearn.manifold import TSNE
import openTSNE
from openTSNE import TSNEEmbedding, TSNE
import numpy as np
import pickle
import matplotlib.pyplot as plt


def main():
    embeddings = None
    with open("embedded_colors.pkl", "rb") as f:
        embeddings = pickle.load(f).data
    embeddings = [embedding.embedding for embedding in embeddings]
    high_dimensional_embeddings = np.array(embeddings)
    affinities = openTSNE.affinity.PerplexityBasedNN(high_dimensional_embeddings)
    init = openTSNE.initialization.pca(high_dimensional_embeddings)

    sample_embedding = openTSNE.TSNE().fit(affinities=affinities, initialization=init)
    sample_embedding: TSNEEmbedding = sample_embedding
    lower_dimensional_embeddings = sample_embedding

    # pickle the TSNEEmbedding model
    with open("tsne_model.pkl", "wb") as f:
        pickle.dump(sample_embedding, f)

    # add the colors (hex, column 2) to the embedded colors
    input_file = open("colors.csv", "r")
    all_lines = input_file.readlines()
    colors = []
    for line in all_lines:
        split_line = line.split(",")
        colors.append(split_line[1])
    input_file.close()

    # add the hex colors to the embedded colors to make points with color for matplotlib, tuples of (x, y, hex)
    points = []

    for i in range(lower_dimensional_embeddings.__len__()):
        points.append(
            (
                lower_dimensional_embeddings[i][0],
                lower_dimensional_embeddings[i][1],
                colors[i].strip(),
            )
        )

    # visualize the embedded colors
    for point in points:
        try:
            plt.scatter(point[0], point[1], color=point[2])
        except:
            print("Invalid color: " + point[2])

    # pickle the points
    with open("points.pkl", "wb") as f:
        pickle.dump(points, f)

    plt.show()


if __name__ == "__main__":
    main()
