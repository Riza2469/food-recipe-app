from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
import os
import pymongo
from pymongo import MongoClient
import requests
from urllib.parse import quote_plus

app = Flask(__name__)
CORS(app)

import ssl
print(ssl.OPENSSL_VERSION)

# MongoDB connection setup
# mongo_conn_str = os.getenv('MONGO_CONNECTION_STRING')
mongo_conn_str = os.getenv('MONGO_CONNECTION_STRING') + "&tls=true&tlsVersion=TLS1.2"
if not mongo_conn_str:
    raise ValueError("MongoDB connection string is not set in environment variables.")
# mongo_client = MongoClient(mongo_conn_str)
mongo_client = MongoClient(mongo_conn_str, tlsAllowInvalidCertificates=True)
mongo_db = mongo_client['recipeDatabase']  # Replace with your database name
recipes_collection = mongo_db['recipes']  # Replace with your collection name

def load_data_from_mongodb():
    recipes_cursor = recipes_collection.find()
    recipes = [recipe for recipe in recipes_cursor]
    return pd.DataFrame(recipes)

def fetch_image_url(query):
    api_key = os.getenv('GOOGLE_API_KEY')
    cse_id = os.getenv('GOOGLE_CSE_ID')

    if not api_key or not cse_id:
        raise ValueError("Google API key or CSE ID is not set in environment variables.")

    search_url = f"https://www.googleapis.com/customsearch/v1?q={quote_plus(query)}&cx={cse_id}&searchType=image&num=1&key={api_key}"

    try:
        response = requests.get(search_url)
        if response.status_code == 200:
            data = response.json()
            if data.get('items'):
                return data['items'][0]['link']
    except Exception as e:
        print(f"Error fetching image URL: {e}")

    return None


@app.route('/cuisines', methods=['GET'])
def get_unique_cuisines():
    print("Fetching cuisines...")
    try:
        print("Loading data from mongodb")
        recipes_df = load_data_from_mongodb()
        print("Data loaded successfully")
        other_cuisines = recipes_df[recipes_df['Type'] == 'Other']['cuisine'].unique().tolist()
        unique_cuisines = ['Indian'] + other_cuisines
        filtered_cuisines = [cuisine for cuisine in unique_cuisines if cuisine != "nan"]
        return jsonify(filtered_cuisines)
    except Exception as e:
        print(f"Error fetching cuisines: {str(e)}")
        return jsonify({'error': f'Failed to fetch cuisines: {str(e)}'}), 500

# @app.route('/cuisines', methods=['GET'])
# def get_unique_cuisines():
#     print("Fetching cuisines...")
#     return jsonify({"message": "Route is working"})


# def recommend_recipes():
#     data = request.get_json()
#     input_ingredients = data.get('input', '')
#     cuisine_types = data.get('cuisine_type', [])
#     if isinstance(cuisine_types, str):
#         cuisine_types = [cuisine_types.lower()]
#     elif not isinstance(cuisine_types, list):
#         cuisine_types = []

#     recipes_df = load_data_from_mongodb()
#     if recipes_df.empty:
#         return jsonify({'error': 'No recipes found'}), 400

#     # Ensure ingredients and cuisine are lowercase for consistent comparison
#     recipes_df['ingredients'] = recipes_df['ingredients'].str.lower()
#     # recipes_df['cuisine'] = recipes_df['cuisine'].str.lower()
#     if cuisine_types:
#         # Make sure to compare with the 'Type' column if that's what you're using
#         filtered_recipes = recipes_df[recipes_df['Type'].str.lower().isin(cuisine_types)]
#     else:
#         # If no cuisine type is specified, consider all recipes
#         filtered_recipes = recipes_df

#     input_ingredients_set = set(ingredient.strip().lower() for ingredient in input_ingredients.split(','))

#     # Filter by cuisine only if cuisine_types is not empty
#     if cuisine_types:
#         filtered_recipes = recipes_df[recipes_df['cuisine'].isin(cuisine_types)]
#     else:
#         # If no cuisine type is specified, consider all recipes
#         filtered_recipes = recipes_df

#     # Then, filter by matching ingredients
#     filtered_recipes['matching_ingredients'] = filtered_recipes['ingredients'].apply(
#         lambda x: sum(ingredient in input_ingredients_set for ingredient in x.split(', '))
#     )

#     # Proceed with TF-IDF, cosine similarities, and combined score calculation...
#     try:
#         tfidf_vectorizer = TfidfVectorizer(stop_words='english')
#         tfidf_matrix = tfidf_vectorizer.fit_transform(filtered_recipes['ingredients'])

#         input_vec = tfidf_vectorizer.transform([input_ingredients])
#         cosine_similarities = linear_kernel(input_vec, tfidf_matrix).flatten()

#         filtered_recipes['similarity_score'] = cosine_similarities
#         filtered_recipes['rating'] = filtered_recipes.get('rating', 0).astype(float)

#         filtered_recipes['combined_score'] = filtered_recipes['matching_ingredients'] * 0.5 + filtered_recipes['similarity_score'] * 0.25 + filtered_recipes['rating'] * 0.25

#         recommendations = filtered_recipes.sort_values(by=['combined_score', 'matching_ingredients'], ascending=False).head(10)
#     except Exception as e:
#         return jsonify({'error': f'Failed to generate recommendations: {str(e)}'}), 500

#     # recommendations_list = [{
#     #     "title": row['recipe_title'],
#     #     "Description": row['description'],
#     #     "Cuisine": row['cuisine'],
#     #     "Course": row['course'],
#     #     "Cook_time": row['cook_time'],
#     #     "Author": row['author'],
#     #     "Ingredients": row['ingredients'],
#     #     "Instructions": row['instructions'],
#     # } for _, row in recommendations.iterrows()]

#     recommendations_list = [{
#         "title": row['recipe_title'],
#         "description": row['description'],
#         "cuisine": row['cuisine'],
#         "course": row['course'],
#         "cook_time": row['cook_time'],
#         "author": row['author'],
#         "ingredients": row['ingredients'],
#         "instructions": row['instructions'],
#         "image_url": row.get('image_url') or fetch_spoonacular_image_url(row['recipe_title']),
#         "rating": row['rating']
#     } for _, row in recommendations.iterrows()]

#     return jsonify({'recommendations': recommendations_list})

def recommend_recipes():
    data = request.get_json()
    input_ingredients = data.get('input', '')
    cuisine_types = data.get('cuisine_type', [])
    if isinstance(cuisine_types, str):
        cuisine_types = [cuisine_types.lower()]
    elif not isinstance(cuisine_types, list):
        cuisine_types = []

    recipes_df = load_data_from_mongodb()
    if recipes_df.empty:
        return jsonify({'error': 'No recipes found'}), 400

    recipes_df['ingredients'] = recipes_df['ingredients'].str.lower()
    if cuisine_types:
        filtered_recipes = recipes_df[recipes_df['Type'].str.lower().isin(cuisine_types)]
    else:
        filtered_recipes = recipes_df

    input_ingredients_set = set(ingredient.strip().lower() for ingredient in input_ingredients.split(','))

    if cuisine_types:
        filtered_recipes = recipes_df[recipes_df['cuisine'].isin(cuisine_types)]
    else:
        filtered_recipes = recipes_df

    filtered_recipes['matching_ingredients'] = filtered_recipes['ingredients'].apply(
        lambda x: sum(ingredient in input_ingredients_set for ingredient in x.split(', '))
    )

    try:
        tfidf_vectorizer = TfidfVectorizer(stop_words='english')
        tfidf_matrix = tfidf_vectorizer.fit_transform(filtered_recipes['ingredients'])
        input_vec = tfidf_vectorizer.transform([input_ingredients])
        cosine_similarities = linear_kernel(input_vec, tfidf_matrix).flatten()
        filtered_recipes['similarity_score'] = cosine_similarities
        filtered_recipes['rating'] = filtered_recipes.get('rating', 0).astype(float)
        filtered_recipes['combined_score'] = filtered_recipes['matching_ingredients'] * 0.5 + filtered_recipes['similarity_score'] * 0.25 + filtered_recipes['rating'] * 0.25
        recommendations = filtered_recipes.sort_values(by=['combined_score', 'matching_ingredients'], ascending=False).head(10)
    except Exception as e:
        return jsonify({'error': f'Failed to generate recommendations: {str(e)}'}), 500

    recommendations_list = []
    for _, row in recommendations.iterrows():
        image_url = row.get('image_url') or fetch_image_url(row['recipe_title'])
        if not image_url:
            # Fallback to a default image or another source if Spoonacular doesn't return an image
            image_url = "https://example.com/default_image.jpg"
        recommendations_list.append({
            "title": row['recipe_title'],
            "description": row['description'],
            "cuisine": row['cuisine'],
            "course": row['course'],
            "cook_time": row['cook_time'],
            "author": row['author'],
            "ingredients": row['ingredients'],
            "instructions": row['instructions'],
            "image_url": image_url,
            "rating": row['rating']
        })

    return jsonify({'recommendations': recommendations_list})


@app.route('/recommend', methods=['POST'])
# def handle_recommend_request():
#     return recommend_recipes()
def handle_recommend_request():
    data = request.json
    print(f"Received data: {data}")
    recommendations = recommend_recipes()
    print(f"Sending back: {recommendations}")
    return recommendations

@app.route('/test', methods=['GET'])
def test():
    return jsonify({"message": "Test route is working!"}), 200

if __name__ == '__main__':
    app.run(debug=True)