# Sentiment-Based Product Recommendation System

A full-stack machine learning web application that delivers product recommendations filtered by customer sentiment using a trained NLP model. Built for the Ebuss e-commerce platform, production-ready for Heroku (or local) deployment with Flask.

***

## Features

- Sentiment analysis on user reviews using TF-IDF and best-performing ML model.
- User-based collaborative filtering for recommendations.
- Top 5 product recommendations, re-ranked by positive sentiment percentage.
- Clean UI with Flask, ready for cloud (Heroku) or local deployment.
- Modular, easily extensible, and reproducible: all code, pickles, and training notebook included.

***

## Folder Structure

```
product-recommendation-flask/
│
├── app.py                # Flask API connecting UI and model
├── model.py              # Loads model pickles, exposes recommend functions
├── requirements.txt      # Python dependencies
├── Procfile              # Heroku config for gunicorn start
├── runtime.txt           # (Optional) For Heroku Python version
│
├── models/               # All required .pkl files for inference
│   ├── sentiment-classification-xg-boost-best-tuned.pkl
│   ├── tfidf_vectorizer.pkl
│   ├── user_final_rating.pkl
│   └── cleaned_data.pkl
│
├── templates/
│   └── index.html        # Web UI for username input and results
│
├── static/
│   └── style.css         # (Optional) Any CSS for frontend
│
├── notebooks/
│   └── SBPRS_Bramhanayaghe_Arumugam.ipynb
│
├── README.md
└── .gitignore
```


***

## Quick Start

1. **Clone this repository:**

```sh
git clone https://github.com/your-username/product-recommendation-flask.git
cd capstoneflask

2. **Create and activate a virtual environment:**

```sh
python -m venv venv
source venv/bin/activate   # On Windows: venv\Scripts\activate
```

3. **Install requirements:**

```sh
pip install -r requirements.txt
```

4. **Run the application locally:**

```sh
python app.py
```

Visit `http://localhost:5000` in your browser.

***

## Heroku Deployment

1. **Login and create an app:**

```sh
heroku login
heroku create <your-app-name>
```

2. **Add, commit, and push everything to Heroku:**

```sh
git add .
git commit -m "Initial commit"
git push heroku main    # or: git push heroku master
```

3. **Open your deployed app in the browser:**

```sh
heroku open
```

*Ensure all `.pkl` files are present in `/models` before deployment.*

***

## How it Works

- User enters their username, clicks "Get Recommendations".
- Backend retrieves top 20 products using collaborative filtering matrix, then re-ranks by analyzing the sentiment in recent reviews.
- The top 5 products with the highest percentage of positive sentiment are returned and displayed.

***

## Model \& Data

- **Sentiment Model:** Best performing NLP classifier (train process in `notebooks/SBPRS_Bramhanayaghe_Arumugam.ipynb`).
- **Recommendation System:** User-item utility matrix (collaborative filtering), persisted as `user_final_rating.pkl`.
- **TF-IDF Vectorizer:** Fitted on training data, transforms review text at prediction time.

***

## Development/Regeneration

- All data cleaning, feature extraction, and model training steps can be found in the notebook (see `/notebooks/`). Re-train and re-pickle if model/data changes.

***

## Author

Your Name ([bramhanayaghe@gmail.com](mailto:bramhanayaghe@email.com))

***