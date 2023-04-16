# Movie recommendation
**Collaboreative filtering approach to give movie recommendations**

**Inputs** pd.DataFrame with: 
* user columns 
* movie rows
* ratings (from 0 to 5) in intersections 

**Outputs** N movie recommendations for specific user.

Guideline to use application:
1. Specify your pandas DataFrame in Recommender class (random values from 0 to 5 by default).
Example: `rec_sys = Recommender(data)`
2. Call recommend_movies function. Specificy user and number of recommendations. Example: `my_recommendation = rec_sys.recommend_movies(user='user_0',
                                                 num_of_recommendations=3)`
3. Get output recommendations in tuples (movie, rate). Example for 'user_0': _[('movie_14', 4.0), ('movie_16', 3.0), ('movie_15', 2.95)]_
