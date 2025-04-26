from flask import Blueprint, request, jsonify
from services.recommendation_service import recommend_movies
import pandas as pd

recommendation_bp = Blueprint('recommendation', __name__)

movies_df = pd.read_csv("data/movies.csv")

@recommendation_bp.route('/recommend', methods=['POST'])
def get_recommendations():
    data = request.get_json()
    user_id = int(data.get("user_id"))
    recommendations = recommend_movies(user_id)
    return jsonify(recommendations)

@recommendation_bp.route('/movies', methods=['GET'])
def get_movies():
    movies = movies_df['title'].tolist()
    return jsonify(movies)