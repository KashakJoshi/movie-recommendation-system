import pandas as pd

movies = []

with open("data/movies.dat", encoding="latin-1") as f:
    for line in f:
        parts = line.strip().split("::")
        movie_id = int(parts[0])
        title = parts[1]
        genres = parts[2]

        movies.append([movie_id, title, genres])

df = pd.DataFrame(movies, columns=["movieId", "title", "genres"])

df.to_csv("api/movies.csv", index=False)

print("movies.csv created successfully")