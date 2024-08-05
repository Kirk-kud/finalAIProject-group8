**Project Title:** Movie Recommendation System Using K-Means Clustering and Cosine Similarities of Plots Within Clusters

**Project Description:**
This project utitlises tools such as the Tfidf vectorizer, which is used for converting text to vectors which can be more easily understood and interpreted by Machine Learning (ML) models. From a dataset of about 46,000 movies, each with a description of their plots and their ratings as listed on the official IMDb website.

**Dataset Used:** https://www.kaggle.com/datasets/samruddhim/imdb-movies-analysis

**How to Application Works:**
The user, upon running the application, is asked for a description of the kind of movie they want to watch. This input is then vectorized behind the scenes, passed to the fit_predict method of the KMeans model. After determining which cluster of movies the user's choice is most likely to be present in, the search area has been narrowed. Now, the prompt is compared with the movies in the designated cluster and up to a maximum of 10 movies may be returned depending on the user's preferences.

**How to Access the Project:**
Currently, it has been deployed on streamlit and can be accessed using the link: https://movie-recommend-ai.streamlit.app/

Alternatively, it could be locally hosted using the .py file also found in this repository. It is implemented using streamlit so it must be installed in the terminal beforehand. All other necessary modules are listed in the 'requirements.txt' file. After all this, the file can be run using the following command:
**streamlit run movie_app.py**

**Credits:**
Project Authors: Welile Dlamini and Kirk Kudoto

Link to YouTube demo video: **{insert video link here}**
