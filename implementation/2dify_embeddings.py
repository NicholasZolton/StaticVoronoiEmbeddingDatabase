# from sklearn.manifold import TSNE
from openTSNE import TSNE
import numpy as np
import pickle
import matplotlib.pyplot as plt


def main():
    embeddings = None
    with open("embedded_colors.pkl", "rb") as f:
        embeddings = pickle.load(f).data
    embeddings = [embedding.embedding for embedding in embeddings]

    # convert to a list of lists of floats
    matrix = np.array(embeddings)

    # create a TSNE model to reduce the dimensions of the embeddings
    tsne = TSNE(n_components=2)
    lower_dimensional_embeddings = tsne.fit(matrix)

    # pickle the TSNE model
    with open("tsne_model.pkl", "wb") as f:
        pickle.dump(tsne, f)

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

    plt.show()


if __name__ == "__main__":
    main()
