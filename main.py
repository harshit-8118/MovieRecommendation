import numpy as np
import pandas as pd
import warnings
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings('ignore')

columns = ['user_id', 'item_id', 'rating', 'timestamp']

''' 
dataset of 100thousand users who have rated movies, 
we are given with user_id, movie_id as item_id, rating to corresponding movie, and timestamp also
=== read README for full reference
'''
user_data = pd.read_csv('MovieRecommendationSystem//MovieLens100k//u.data', sep='\t', names=columns)

# set panda to display full table
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)

# print(user_data.head())
# print(user_data.shape)

# print(user_data['user_id'].nunique())
# print(user_data['item_id'].nunique())

movie_columns = ["item_id", "title", "release date", "video release date","IMDb URL", "unknown", "Action", "Adventure", "Animation","Children's", "Comedy", "Crime", "Documentary", "Drama", "Fantasy","Film-Noir", "Horror", "Musical", "Mystery", "Romance", "Sci-Fi","Thriller", "War", "Western"]
movie_data = pd.read_csv('MovieRecommendationSystem/MovieLens100k/u.item', sep='\|', header=None, encoding='ISO-8859-1', names=movie_columns)
movie_data = movie_data.drop(columns=['video release date'])

# print(movie_data)
# print(movie_data.shape)

movie_data = movie_data[['item_id', 'title']]
# print(movie_data)

'''
Merging two table on (item_id == movie_id) 
from [user_data, movie_data]
'''
data = pd.merge(user_data, movie_data, on='item_id')

# final data to process
'''
user_id, item_id, rating, timestamp, title
'''
# print(data)


'''
Exploratory Data Analysis
'''

''' no use for Average of user_id, item_id, and timestamp '''
# print(data.groupby('title').mean()['rating'].sort_values(ascending=False))

''' most watched movies '''
# print(data.groupby('title').count()['rating'].sort_values(ascending=False))


'''
New ratings table 
columns = ['title', 'rating', 'num of ratings']
'''
ratings = pd.DataFrame(data.groupby('title').mean()['rating'])
# print(ratings.head())
ratings['num of ratings'] = pd.DataFrame(data.groupby('title').count()['user_id'])
# print(ratings.head())

# ratings = ratings.sort_values(by='rating', ascending=False)
# print(ratings)

''' discard movies which are highly rated by (less no of people) 
and discard movies which are low rated 
and also less rated movies
'''

plt.figure(figsize=(10, 6))

''' freq of movies Vs. No of ratings '''
# plt.hist(ratings['num of ratings'], bins=80)
# plt.show()

''' average No of movies{_Y} Vs rating{_X} '''
# plt.hist(ratings['rating'], bins=80)
# plt.show()

''' rating{_X} vs no of ratings{_Y}'''
# sns.jointplot(x='rating', y='num of ratings', data=ratings)
# plt.show() 


''' Movie Recommendation '''
# print(data)
moviemat = data.pivot_table(index='user_id', columns='title', values='rating')
# print(moviemat)

# print(ratings.sort_values('num of ratings', ascending=False).head())


def predictMovies(movie_name):
    movieUserRatings = moviemat[movie_name]
    # print(movieUserRatings)

    similarToMovieUser = moviemat.corrwith(movieUserRatings)
    # print(similarTomovieUser)

    corrMovieUser = pd.DataFrame(similarToMovieUser, columns=['Correlation'])
    # print(corrMovieUser)

    corrMovieUser.dropna(inplace=True)
    # print(corrMovieUser)

    # corrMovieUser.sort_values('Correlation', ascending=False, inplace=True)
    ''' highly correlated movies may not be good choice
    because one movie rated by 300 people as 4.0 average
    and other movie rated as 4.0 by 10 peoples shows perfect correlated but it should not be perfect match.

    we will consider if both movies are rated above threshold number of people
    '''
    # print(corrMovieUser)

    corrMovieUser = corrMovieUser.join(ratings['num of ratings'])
    # print(corrMovieUser.head())

    predictedMovies = corrMovieUser[corrMovieUser['num of ratings'] > 100].sort_values(by='Correlation', ascending=False)
    # print(predictedMovies)
    return predictedMovies
    
if __name__ == '__main__':
    movies = ['Star Wars (1977)', 'Titanic (1997)', 'Underneath, The (1995)']
    for movie in movies:
        predictions = predictMovies(movie.strip())
        print(predictions.head(10))







 




