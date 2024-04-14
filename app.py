import streamlit as st
import torch
import pandas as pd
import settings as s
from utils import load_json, Recommender, highlight_clicked_rows

import json


# Load pre-trained model
recommender = Recommender(s.impressions_path, s.user_embedding_path, s.news_embedding_path, s.user_map_path, True)
recommender.load_state_dict(torch.load(s.model_path))

user_history = pd.read_parquet(s.user_history_path).set_index('user_id').to_dict()['history']

impressions = pd.read_parquet(s.impressions_path).drop_duplicates()

news_emb = pd.read_parquet(s.news_embedding_path)
news_data = pd.read_parquet(s.news_data_path)
user_map = load_json(s.user_map_path)

user_map = dict(list(user_map.items()))

# Sidebar
st.sidebar.title("Input parameters and user details")

user_id = st.sidebar.selectbox("Select user ID:", list(user_map.keys()), index=6)
if user_id:
    news_impressions = impressions.loc[impressions['user_id'] == user_id, 'news_id'].drop_duplicates().tolist()
    news_clicks = impressions.loc[impressions['user_id'] == user_id, ['news_id', 'target']].rename(columns={'target':'is_clicked'})
    num_recommendations = st.sidebar.number_input("Number of Recommendations", min_value=2, max_value=len(news_impressions), value=min(5, len(news_impressions)))

    st.sidebar.subheader("User History")
    st.sidebar.dataframe(news_data[news_data['news_id'].isin(user_history[user_id])], hide_index=True)
    st.sidebar.subheader("News impressions")
    st.sidebar.dataframe(news_data[news_data['news_id'].isin(news_impressions)], hide_index=True)
    

# Main content
st.title("News Recommender System")
st.write("Welcome to the news recommender system! In this application, you can explore personalized recommendations based on users preferences and past interactions.")

st.header("Recommendations Settings")
st.write("**Number of Recommendations**: Choose the number of recommendations you'd like to see. You can select a minimum of 2 and a maximum of the number of impressions you've made.")

st.header("Understanding the Recommendations")
st.write("**Highlighted Articles**: Articles highlighted in light green represent the ones the user actually clicked on.")
st.write("**News impressions**: These are the articles you've actually seen. We optimize the ordering of these articles to maximize user engagement on the platform. In real-life scenarios, we aim to present the most relevant articles to enhance readers browsing experience.")

if user_id:
    recommended_news = recommender.recommend(user_id, news_impressions, len(news_impressions), 'cpu')
    recommendations = pd.DataFrame(recommended_news, columns=['news_id']).merge(news_data, 'left', 'news_id')
    recommendations = recommendations.merge(news_clicks, 'left', 'news_id')

    st.subheader("Ordered recommendations")
    st.dataframe(recommendations.head(num_recommendations).style.apply(highlight_clicked_rows, axis=1), hide_index=True)
