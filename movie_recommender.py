import streamlit as st
import joblib
import numpy as np
import pandas as pd
import gdown

# URL AND GDOWN
model_url = 'https://drive.google.com/uc?export=download&id=1tIsMTq7T4N3c2PcqYTJKW1h_UWNj26-o'
gdown.cached_download(model_url, 'kmeans_model.joblib')

vectorizer_url = 'https://drive.google.com/uc?export=download&id=15C8_p_rEx4bipwHT-ldWP5ihtT8I90Ys'
gdown.cached_download(vectorizer_url, 'tfidf_vectorizer.joblib')

df_url = 'https://drive.google.com/uc?export=download&id=1T-OKMXWWjwU12Qjy3V-sPtk2WCbewgVl'
gdown.cached_download(df_url, 'dataframe.joblib')

# Load the saved models and data
@st.cache_resource
def load_models():
    model = joblib.load('kmeans_model.joblib')
    vectorizer = joblib.load('tfidf_vectorizer.joblib')
    df = joblib.load('dataframe.joblib')
    return model, vectorizer, df

model, vectorizer, df = load_models()
poster_df = joblib.load('poster_df.joblib')

def recommend_movies(description, top_n=5):
    user_input_vector = vectorizer.transform([description])
    cluster_label = model.predict(user_input_vector)[0]
    cluster_movies = df[df["Cluster"] == cluster_label].copy()
    tokenized_plots = [x.toarray().flatten() for x in cluster_movies['tokenized_plot']]
    cluster_movies['tokenized_plot'] = tokenized_plots

    # Select top_n movies randomly if there are more movies in the cluster than top_n
    if len(cluster_movies) > top_n:
        cluster_movies = cluster_movies.sample(n=top_n)

    recommended_movies = cluster_movies[['title', 'plot', 'imdbRating', 'genre']]
    return recommended_movies

st.title('Perfect Movie for the Weekend')

user_input = st.text_area("Describe the movie you want to watch: ")
num_recommendations = st.number_input("How many recommendations do you want:", min_value=1, max_value=20, value=5, step=1)

if st.button('Recommend Movies'):
    if user_input:
        with st.spinner('Finding recommendations...'):
            recommendations = recommend_movies(user_input, top_n=num_recommendations)
            st.subheader("Recommended Movies:")
            for i, (_, movie) in enumerate(recommendations.iterrows(), 1):
                movie_title = movie['title']
                poster_url = poster_df[poster_df['title'] == movie_title]['poster'].iloc[0] if not poster_df[poster_df['title'] == movie_title].empty else None

                if poster_url:
                    st.image(poster_url, use_column_width=True)
                else:
                    st.write("No poster available for this movie.")
                    
                st.write(f"{i}. {movie['title']}")
                with st.expander("See Official imdB Rating"):
                    st.write(movie['imdbRating'])
                with st.expander("See genre"):
                    st.write(movie['genre'])
                with st.expander("See plot"):
                    st.write(movie['plot'])
    else:
        st.warning("Please enter a movie description.")
