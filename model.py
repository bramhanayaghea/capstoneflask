import pickle
import pandas as pd
import numpy as np

def load_models():
    """Load all required models and data"""
    with open('models/sentiment-classification-xg-boost-best-tuned.pkl', 'rb') as f:
        sentiment_model = pickle.load(f)
    with open('models/tfidf-vectorizer.pkl', 'rb') as f:
        vectorizer = pickle.load(f)
    with open('models/user_final_rating.pkl', 'rb') as f:
        user_final_rating = pickle.load(f)
    with open('models/cleaned-data.pkl', 'rb') as f:
        df_cleaned = pickle.load(f)
    return sentiment_model, vectorizer, user_final_rating, df_cleaned

def get_top_20_recommendations(username, user_final_rating):
    """Get top 20 product recommendations for a user"""
    if username not in user_final_rating.index:
        return None, f"User '{username}' not found in the database."
    user_ratings = user_final_rating.loc[username].sort_values(ascending=False)
    top_20_products = user_ratings.head(20)
    return list(top_20_products.index), None

def calculate_sentiment_percentage(product_name, df_cleaned, sentiment_model, vectorizer):
    product_reviews = df_cleaned[df_cleaned['name'] == product_name].copy()
    if product_reviews.empty:
        return 0.0
    if 'reviews_title' not in product_reviews.columns:
        product_reviews['reviews_title'] = ''
    if 'reviews_text' not in product_reviews.columns:
        product_reviews['reviews_text'] = ''
    # Concatenate and drop completely empty rows
    review_texts = (
        product_reviews['reviews_title'].fillna('') + ' ' +
        product_reviews['reviews_text'].fillna('')
    ).str.strip()
    # Filter out empty reviews
    non_empty_texts = review_texts[review_texts != '']
    if non_empty_texts.empty:
        return 0.0
    X_reviews = vectorizer.transform(non_empty_texts)
    predictions = sentiment_model.predict(X_reviews)
    positive_count = sum(pred == 'Positive' for pred in predictions)
    positive_percentage = (positive_count / len(predictions)) * 100 if len(predictions) > 0 else 0.0
    return positive_percentage


def recommend_top_5_products(username):
    """Get top 5 product recommendations for a given user, sorted by positive sentiment"""
    sentiment_model, vectorizer, user_final_rating, df_cleaned = load_models()
    top_20_products, error = get_top_20_recommendations(username, user_final_rating)
    if error or top_20_products is None:
        return None, error
    # Calculate sentiment percentage for each product
    product_sentiments = []
    for product in top_20_products:
        sentiment_pct = calculate_sentiment_percentage(product, df_cleaned, sentiment_model, vectorizer)
        product_sentiments.append({
            'product_name': product,
            'positive_sentiment_percentage': sentiment_pct
        })
    product_sentiments_df = pd.DataFrame(product_sentiments)
    if product_sentiments_df.empty:
        return None, "No products found to recommend."
    top_5_products = product_sentiments_df.nlargest(5, 'positive_sentiment_percentage')
    return top_5_products, None
