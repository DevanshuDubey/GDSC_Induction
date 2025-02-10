import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors

movies = pd.read_csv("movies.csv")
ratings = pd.read_csv("ratings.csv")
tags = pd.read_csv("tags.csv")

tag_data = tags.groupby("movieId")["tag"].apply(lambda x: " ".join(x)).reset_index()
movies = movies.merge(tag_data, on="movieId", how="left").fillna("")

vectorizer = TfidfVectorizer(stop_words="english")
tfidf_matrix = vectorizer.fit_transform(movies["tag"])
content_sim = cosine_similarity(tfidf_matrix)

rating_matrix = ratings.pivot(index="movieId", columns="userId", values="rating").fillna(0)
csr_ratings = csr_matrix(rating_matrix)
knn_model = NearestNeighbors(metric="cosine", algorithm="brute")
knn_model.fit(csr_ratings)

def content_based_recommendations(movie_title, top_n=10):
    idx = movies[movies["title"].str.contains(movie_title, case=False, na=False)].index[0]
    scores = list(enumerate(content_sim[idx]))
    scores = sorted(scores, key=lambda x: x[1], reverse=True)[1:top_n+1]
    movie_indices = [i[0] for i in scores]
    return movies.iloc[movie_indices][["title"]]

def knn_user_recommendations(user_id, top_n=10):
    if user_id not in ratings["userId"].unique():
        return pd.DataFrame(["User not found. Try another ID."])
    
    user_ratings = ratings[ratings["userId"] == user_id].sort_values(by="rating", ascending=False)
    if user_ratings.empty:
        return pd.DataFrame(["No ratings found for this user."])
    
    top_movie_id = user_ratings.iloc[0]["movieId"]
    movie_idx = rating_matrix.index.get_loc(top_movie_id)
    distances, indices = knn_model.kneighbors(csr_ratings[movie_idx], n_neighbors=top_n+1)
    recommended_indices = indices.flatten()[1:]
    return movies.iloc[recommended_indices][["title"]]

def hybrid_recommendation(movie_title, user_id, top_n=5):
    content_recs = content_based_recommendations(movie_title, top_n)
    knn_user_recs = knn_user_recommendations(user_id, top_n)
    hybrid_recs = pd.concat([content_recs.head(top_n), knn_user_recs.head(top_n)]).drop_duplicates()
    return hybrid_recs

st.title("Movie Recommendation System (KNN-Based)")
user_id = st.number_input("Enter User ID:", min_value=1, step=1)
user_input = st.text_input("Enter a movie title:")
if user_input:
    st.subheader("Content-Based Recommendations")
    st.write(content_based_recommendations(user_input))
    
    st.subheader("KNN Collaborative Filtering (User-Based)")
    st.write(knn_user_recommendations(user_id))
    
    st.subheader("Hybrid Recommendations")
    st.write(hybrid_recommendation(user_input, user_id))
