import streamlit as st
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd

model_path = "/workspaces/Webapp-Streamlit-Julian-Cerro/models/knn_neighbors-5_algorithm-brute_metric-cosine.sav"
model = pickle.load(open(model_path, "rb"))

#load clean_data.csv
data_path = "/workspaces/Webapp-Streamlit-Julian-Cerro/src/clean_data.csv"  
data = pd.read_csv(data_path)

#Train vectorizer with "tags" columns data
vectorizer = TfidfVectorizer()
vectorizer.fit(data["tags"])

st.title("Movie recommendation")
st.markdown("""
This application uses a KNN model trained with TF-IDF to find similar movies based in your preferences.
""")

#Enter "tags" text
user_input = st.text_area("Enter tags (keywords):", "example: adventure, action, science fiction")

#Button to predict
if st.button("Find recommendations"):
    try:
        user_vector = vectorizer.transform([user_input])
        #Find recomm
        distances, indices = model.kneighbors(user_vector, n_neighbors=5)
        st.write("### Recommended movies:")
        for i in indices[0]:
            st.write(data.iloc[i]['title'])
    except ValueError as e:
        st.error("Error: make sure to imput valid data.")
