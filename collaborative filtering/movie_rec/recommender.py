import pandas as pd
import numpy as np
from typing import List, Tuple
from sklearn.neighbors import NearestNeighbors
import logging

from config import Config

logger = logging.getLogger("RecSys")


class Recommender(Config):
    def __init__(self, data: pd.DataFrame = None):
        super().__init__()
        if data is None:
            self.data = self._generate_data(self.num_users, self.num_films)
        else:
            self.data = data

    @staticmethod
    def _generate_data(num_users: int, num_films: int) -> pd.DataFrame:
        """
        Generates pandas table with "num_users" columns and "num_films" rows
        :param num_users: number of users in the table
        :param num_films: number of films in the table
        :return: Pandas DataFrame
        """
        columns = [f"user_{x}" for x in range(0, num_users)]
        index = [f"movie_{x}" for x in range(0, num_films)]

        df = pd.DataFrame(data=np.random.randint(0, 5, size=(num_users, num_films)), columns=columns, index=index)
        logger.info(f"Successfully created table with {num_users} users and {num_films} films.")

        return df

    def _find_distances_and_indices_with_knn(self):
        """
        Determines similarity distances between objects and finds their distances
        :return: List of distances between objects and their indices
        """
        knn = NearestNeighbors(metric=self.metric, algorithm=self.algorithm)
        knn.fit(self.data.values)
        self.distances, self.indices = knn.kneighbors(self.data.values, n_neighbors=self.n_neighbors)

    def _predict_ratings(self) -> None:
        """
        Predicts ratings for all missing rates (for all zeros)
        :return: None
        """
        if self.distances is None or self.indices is None:
            logger.error("Unable to utilize knn to find distances between objects")
            return

        self.predicted_data = self.data.copy()
        # for each user
        for user_index in range(0, len(self.data.columns.tolist())):
            # for each film
            self._predict_row(user_index)

    def _predict_row(self, user_index) -> None:
        """
        Predicts missing ratings for one movie
        :param user_index: index of user
        :return: None
        """
        for row, movie in enumerate(self.data.index):
            # if there is no rate
            if self.data.iloc[row, user_index] == 0:
                sim_movies = self.indices[row].tolist()
                movie_distances = self.distances[row].tolist()

                if row in sim_movies:
                    id_movie = sim_movies.index(row)
                    sim_movies.remove(row)
                    movie_distances.pop(id_movie)
                else:
                    sim_movies = sim_movies[:self.n_neighbors - 1]
                    movie_distances = movie_distances[:self.n_neighbors - 1]

                movie_similarity = [1 - x for x in movie_distances]
                movie_similarity_copy = movie_similarity.copy()

                nominator = 0

                # for each similar movie
                for s in range(0, len(movie_similarity)):
                    if self.data.iloc[sim_movies[s], user_index] == 0:

                        if len(movie_similarity_copy) == (self.n_neighbors - 1):
                            movie_similarity_copy.pop(s)
                        else:
                            movie_similarity_copy.pop(s - (len(movie_similarity) - len(movie_similarity_copy)))

                    else:
                        nominator = nominator + movie_similarity[s] * self.data.iloc[sim_movies[s], user_index]

                if len(movie_similarity_copy) > 0 and sum(movie_similarity_copy) > 0:
                    predict = nominator / sum(movie_similarity_copy)
                else:
                    predict = 0

                self.predicted_data.iloc[row, user_index] = round(predict, 2)

    def recommend_movies(self, user: str, num_of_recommendations: int) -> List[Tuple[str, int]]:
        """
        Gives n movies recommendations to specific user
        :param user: username
        :param num_of_recommendations: number of movies to recommend
        :return: List of recommendations
        """
        recommendations = []
        self._find_distances_and_indices_with_knn()
        self._predict_ratings()

        for m in self.data[self.data[user] == 0].index.tolist():
            index_df = self.data.index.tolist().index(m)
            prediction = self.predicted_data.iloc[index_df, self.data.columns.tolist().index(user)]
            movie = self.data.index.tolist()[index_df]
            recommendations.append((movie, prediction))

        sorted_recommendations = sorted(recommendations, key=lambda x: x[1], reverse=True)

        return sorted_recommendations[:num_of_recommendations]
