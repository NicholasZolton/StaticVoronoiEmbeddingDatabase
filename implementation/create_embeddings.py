import shapely
from shapely.geometry import Point
import random
from openai import OpenAI
import pickle


def main():

    # open the "colors.csv" file and get all the color names (first column)
    input_file = open("colors.csv", "r")
    all_lines = input_file.readlines()
    colors = []
    for line in all_lines:
        split_line = line.split(",")
        colors.append(split_line[0])
    input_file.close()

    # prefix all the colors with "color: "
    for i in range(colors.__len__()):
        colors[i] = "color: " + colors[i]

    # embed all the colors into 2 dimensional space

    client = OpenAI()

    embedded_colors = client.embeddings.create(
        model="text-embedding-3-small", input=colors, dimensions=2
    )

    # pickle the embedded colors
    with open("embedded_colors.pkl", "wb") as f:
        pickle.dump(embedded_colors, f)

    # print the embedded colors
    print(embedded_colors)


if __name__ == "__main__":
    main()
