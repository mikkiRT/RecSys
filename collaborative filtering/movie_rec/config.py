class Config:
    def __init__(self,
                 n_neighbors: int = 3,
                 num_users: int = 20,
                 num_films: int = 20,
                 metric: str = 'cosine',
                 algorithm: str = 'brute',):
        """
        Configuration file for Recommendation system
        :param n_neighbors: number of neighbors in knn
        :param num_users: number of users in generated table
        :param num_films: number of movies in generated table
        :param metric: metric in searching distances between objects
        :param algorithm: algorithm in searching distances between objects
        """
        self.n_neighbors = n_neighbors
        self.num_users = num_users
        self.num_films = num_films
        self.metric = metric
        self.algorithm = algorithm
