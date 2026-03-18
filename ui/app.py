import streamlit as st
import requests


# Render deployment API URL
API_URL = "https://your-api.onrender.com/recommend"
#API_URL = "http://127.0.0.1:8000/recommend"


st.set_page_config(
    page_title="Movie Recommender",
    page_icon="🎬",
    layout="wide"
)

st.title("🎬 Movie Recommendation System")

st.markdown(
    "Get personalized movie recommendations using Machine Learning"
)

user_id = st.number_input(
    "Enter User ID",
    min_value=1,
    step=1
)

if st.button("Get Recommendations"):

    try:

        response = requests.get(
            API_URL,
            params={"user_id": user_id}
        )

        if response.status_code == 200:

            data = response.json()

            st.subheader("⭐ Top Recommended Movies")

            for movie in data["recommendations"]:

                rating = movie["predicted_rating"]

                st.write(
                    f"🎬 {movie['title']} ⭐ Rating: {rating}"
                )

        else:
            st.error("API Error")

    except Exception as e:
        st.error(f"Connection Error: {e}")