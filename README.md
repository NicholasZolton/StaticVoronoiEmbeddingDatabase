# Introduction

This is a repository dedicated to my final project in Computational Geometry, creating a database with Voronoi Diagrams as the backend to allow for efficient nearest neighbor queries relevant to AI embeddings in extremely low dimensions.

# Installation

Start by creating a Python virtual environment:

`python -m virtualenv ./.venv`

Activate the virtual environment:

`source ./.venv/bin/activate`

Install the required packages:

`pip install -r requirements.txt`

# Usage

Since this is meant to be used as a library but also exists as a standalone project/report, the usage is split into two parts within the `implementation` directory:

First, there's the VoronoiDatabase, the main class and final product of this project. It can be used as a standalone library to create a Voronoi Database and query it for nearest neighbors. The VoronoiDatabase class is located in the `implementation/VoronoiDatabase.py` file.

Second, there's all of the other files in `implementation`. While these are messy and not necessarily useful, I decided to keep them in the repository to show the process I went through to create the VoronoiDatabase class.

# Report

The report for this project can be found in the `report` directory. It is a LaTeX document that explains the project, the process, and the results in detail. `report/ref.bib` contains all of the references used in the report, and `report/main.pdf` is the compiled report.