from recommender import Recommender


def main():
    rec_sys = Recommender()
    print(rec_sys.data)

    my_recommendation = rec_sys.recommend_movies(user='user_0',
                                                 num_of_recommendations=3)
    print(rec_sys.predicted_data)

    print(my_recommendation)  # [('movie_14', 4.0), ('movie_16', 3.0), ('movie_15', 2.95)]


if __name__ == "__main__":
    main()
