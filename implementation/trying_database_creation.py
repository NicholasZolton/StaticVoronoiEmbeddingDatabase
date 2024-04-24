from VoronoiDatabase import NDVoronoiDatabase


def main():
    # open the "colors.csv" file and get all the color names (first column)
    input_file = open("colors.csv", "r")
    all_lines = input_file.readlines()
    colors = []
    color_hexes = []
    for line in all_lines:
        split_line = line.split(",")
        colors.append(split_line[0])
        color_hexes.append(split_line[1].strip())
    input_file.close()

    # create the NDVoronoiDatabase
    voronoi_database = NDVoronoiDatabase(colors, colors=color_hexes)
    voronoi_database.save("voronoi_database.pkl")

    # print the NDVoronoiDatabase
    print(voronoi_database)


if __name__ == "__main__":
    main()
