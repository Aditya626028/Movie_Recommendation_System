
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

data = {
    'User': ['B','B','B','C','C','C','D','D','D','E','E','E','F','F','F','G','G','G','H','H','H'],
    'Movie': ['Titanic','Avatar','Inception',
              'Titanic','Inception','Avengers',
              'Avatar','Inception','Avengers',
              'Titanic','Avengers','Avatar',
              'Inception','Avengers','Titanic',
              'Titanic','Inception','Avatar',
              'Avengers','Titanic','Inception'],
    'Rating': [4,5,2,5,4,3,4,3,5,5,3,4,4,5,3,3,4,5,4,5,4]
}

df = pd.DataFrame(data)

pivot_table = df.pivot_table(index='User', columns='Movie', values='Rating').fillna(0)

user_similarity = cosine_similarity(pivot_table)
user_similarity_df = pd.DataFrame(user_similarity, index=pivot_table.index, columns=pivot_table.index)

def recommend_movies(user, top_n=5):
    similar_users = user_similarity_df[user].sort_values(ascending=False)[1:]  # exclude self
    recommended_movies = pd.Series(dtype=float)
    
    for other_user, similarity in similar_users.items():
        other_ratings = pivot_table.loc[other_user]
        recommended_movies = recommended_movies.add(other_ratings * similarity, fill_value=0)
    
    already_rated = pivot_table.loc[user]
    recommended_movies = recommended_movies[already_rated == 0]
    
    return recommended_movies.sort_values(ascending=False).head(top_n)

user_to_recommend = 'B'  
recommended = recommend_movies(user_to_recommend, top_n=5)

print(f"Top 5 recommended movies for user {user_to_recommend}:")
for idx, score in recommended.items():
    print(f"{idx}: Predicted rating {score:.2f}")
