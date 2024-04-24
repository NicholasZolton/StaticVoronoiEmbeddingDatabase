from VoronoiDatabase import NDVoronoiDatabase
import matplotlib.pyplot as plt


def main():
    vordb = NDVoronoiDatabase.load("voronoi_database.pkl")

    # plot = vordb.visualize_voronoi(plt=plt)
    # plot.show()

    print(vordb.query_input("the lime fruit", visualize=True))


if __name__ == "__main__":
    main()
