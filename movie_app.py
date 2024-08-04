import streamlit as st
import joblib
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Load the saved models and data
@st.cache_resource
def load_models():
    model = joblib.load('kmeans_model.joblib')
    vectorizer = joblib.load('tfidf_vectorizer.joblib')
    df = joblib.load('dataframe.joblib')
    return model, vectorizer, df

model, vectorizer, df = load_models()

def recommend_movies(description, top_n=5):
    user_input_vector = vectorizer.transform([description])
    cluster_label = model.predict(user_input_vector)[0]
    cluster_movies = df[df["Cluster"] == cluster_label].copy()
    tokenized_plots = [x.toarray().flatten() for x in cluster_movies['tokenized_plot']]
    cluster_movies['tokenized_plot'] = tokenized_plots
    similarities = cosine_similarity(user_input_vector, np.vstack(cluster_movies['tokenized_plot'])).flatten()
    top_indices = similarities.argsort()[-top_n:][::-1]
    recommended_movies = cluster_movies.iloc[top_indices][['title', 'plot']]
    return recommended_movies

st.title('Movie Recommender')

user_input = st.text_area("Enter a movie description:")
if st.button('Recommend Movies'):
    if user_input:
        with st.spinner('Finding recommendations...'):
            recommendations = recommend_movies(user_input)
            st.subheader("Recommended Movies:")
            for i, (_, movie) in enumerate(recommendations.iterrows(), 1):
                st.write(f"{i}. {movie['title']}")
                with st.expander("See plot"):
                    st.write(movie['plot'])
    else:
        st.warning("Please enter a movie description.")
