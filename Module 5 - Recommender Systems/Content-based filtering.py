# Explore Content-based recommendation systems and implement a simpler version...
# ...of one using Python and the Pandas library
# Import needed libraries
import pandas as pd
from math import sqrt
import numpy as np
import matplotlib.pyplot as plt

# Read files into Dataframes
# Movie information
movies_df = pd.read_csv("movies.csv")
# user information
ratings_df = pd.read_csv("ratings.csv")

# Remove the year from the 'title' column by using pandas's replacement function and store in a new 'year' column
# Using regular expressions to find a year stored between parentheses (Ex: Toy Story (1995); Grumpier Old Men (1995))
# Specify the parantheses so we do not conflict with movies which have years in their titles
movies_df['year'] = movies_df.title.str.extract('(\(\d\d\d\d\))', expand=False)
# Remove parentheses
movies_df['year'] = movies_df.year.str.extract('(\d\d\d\d)', expand=False)
# Remove the year from the 'title' column
movies_df['title'] = movies_df.title.str.replace('(\(\d\d\d\d\))', '')
# Ger rid of any ending whitespace characters that may have appeared
movies_df['title'] = movies_df['title'].apply(lambda x: x.strip())

# Split the Genres column into a list of Genres to simplify future use
# Every genre is separated by a | so we simply have to call a split function on |
movies_df["genres"] = movies_df.genres.str.split('|')

# Use One Hot Encoding technique to convert the list of genres to a vector where...
# ...each column corresponds to one possible value of the feature
# Copy the movie dataframe into a new one
movies_with_genres_df = movies_df.copy()

# For every row in the dataframe, iterate through the list of genres and place a 1 into the corresponding column
for index, row in movies_df.iterrows():
    for genre in row['genres']:
        movies_with_genres_df.at[index, genre] = 1
# Fill in the NaN values with 0 to show that a movie doesn't have that column's genre
movies_with_genres_df = movies_with_genres_df.fillna(0)

# Drop the timestamp column because we do not need it
# 'Drop' function removes a specific row or column from a dataframe
ratings_df = ratings_df.drop('timestamp', 1)

# Create an input user to recommend movies
userInput = [
            {'title':'Breakfast Club, The', 'rating':5},
            {'title':'Toy Story', 'rating':3.5},
            {'title':'Jumanji', 'rating':2},
            {'title':"Pulp Fiction", 'rating':5},
            {'title':'Akira', 'rating':4.5}
         ]
input_movies = pd.DataFrame(userInput)

# Add movieId to input user
# filter out the movies by title
inputId = movies_df[movies_df['title'].isin(input_movies['title'].tolist())]
# merge inputId into input_movies
input_movies = pd.merge(inputId, input_movies)
# drop information we will not use from the input dataframe
input_movies = input_movies.drop('genres', 1).drop('year', 1)

# Filter out the movies from the input
user_movies = movies_with_genres_df[movies_with_genres_df['movieId'].isin(input_movies['movieId'].tolist())]

# Reset the index to avoid future issues
user_movies = user_movies.reset_index(drop=True)
# Drop unnecessary issues due to save memory and to avoid issues
user_genre_table = user_movies.drop('movieId', 1).drop('title', 1).drop('genres', 1).drop('year', 1)

# Learn the input's preference
# Turn each genre into weights. We can do this by using the input's reviews and multiply them with the input's genre table...
# ...and then sum up the resulting table by column
# Use dot product to get weights
user_profile = user_genre_table.transpose().dot(input_movies['rating'])

# Extract the genre table from the original dataframe
# Get the genres of every movie in our original dataframe
genre_table = movies_with_genres_df.set_index(movies_with_genres_df['movieId'])
# Drop some unnecessary information
genre_table = genre_table.drop('movieId', 1).drop('title', 1).drop('genres', 1).drop('year', 1)

# Now we have the input profile and the complete list of movies and their genres in hand.
# We will take the average weights of every movie based on the input profile and recommend the top twenty movies that most satisfy the user
# Multiply the genres by the weights and then take the average weights
recommendation_table = ((genre_table * user_profile).sum(axis=1))/(user_profile.sum())

# Sort our recommendations in descending order
recommendation_table = recommendation_table.sort_values(ascending=False)

# The final recommedation table
print(movies_df.loc[movies_df['movieId'].isin(recommendation_table.head(20).keys())])