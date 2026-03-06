import streamlit as st
import requests

API_URL = "http://127.0.0.1:8000/recommend"

st.set_page_config(page_title="Movie Recommender", page_icon="🎬", layout="wide")

st.title("🎬 Movie Recommendation System")

st.markdown("Get personalized movie recommendations using Machine Learning")

user_id = st.number_input("Enter User ID", min_value=1, step=1)

if st.button("Get Recommendations"):

    response = requests.get(API_URL, params={"user_id": user_id})

    if response.status_code == 200:

        data = response.json()

        st.subheader("⭐ Top Recommended Movies")

        cols = st.columns(2)

        for i, movie in enumerate(data["recommendations"]):

            with cols[i % 2]:

                rating = movie["predicted_rating"]

                stars = "⭐" * int(round(rating))

                st.markdown(
                    f"""
                    ### 🎥 {movie['movie']}
                    
                    **Predicted Rating:** {rating:.2f}

                    {stars}

                    ---
                    """
                )

    else:
        st.error("API Error")