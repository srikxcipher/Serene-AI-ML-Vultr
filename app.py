# app.py
from flask import Flask, jsonify, request
from flask_cors import CORS
import random
import logging
import pandas as pd
import csv
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.cluster import KMeans
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import json
import numpy as np

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Custom JSON Encoder for handling NumPy types
class CustomJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        return super().default(obj)

# Set the custom JSON encoder
app.json_encoder = CustomJSONEncoder

# Configure logging
logging.basicConfig(filename='app.log', level=logging.INFO, 
                    format='%(asctime)s %(levelname)s: %(message)s')

# Load models and sentiment analyzer
sentiment_analyzer = SentimentIntensityAnalyzer()

# Load habit recommendation data
habit_data = pd.read_csv('habit_data.csv')

# Preprocess the data
# Encode categorical variables
le_exercise = LabelEncoder()
le_social_media = LabelEncoder()
le_stress = LabelEncoder()
le_mindfulness = LabelEncoder()

habit_data['exercise_frequency'] = le_exercise.fit_transform(habit_data['exercise_frequency'])
habit_data['social_media_hours'] = le_social_media.fit_transform(habit_data['social_media_hours'])
habit_data['stress_level'] = le_stress.fit_transform(habit_data['stress_level'])
habit_data['mindfulness_frequency'] = le_mindfulness.fit_transform(habit_data['mindfulness_frequency'])

# Features and target variable
X = habit_data.drop('recommended_habit', axis=1)
y = habit_data['recommended_habit']

# Create and train the Decision Tree model
model = DecisionTreeClassifier(random_state=42)
model.fit(X, y)

# Load music data
music_data = pd.read_csv('music_data.csv')

# Prepare response pools for chatbot
positive_responses = []
neutral_responses = []
negative_responses = []

# Load responses from CSV for the chatbot
def load_responses(filename):
    global positive_responses, neutral_responses, negative_responses
    with open(filename, mode='r', encoding='utf-8') as file:
        reader = csv.DictReader(file)
        for row in reader:
            if row['Sentiment'] == 'Positive':
                positive_responses.append(row['Response'])
            elif row['Sentiment'] == 'Neutral':
                neutral_responses.append(row['Response'])
            elif row['Sentiment'] == 'Negative':
                negative_responses.append(row['Response'])

# Load initial responses
load_responses('responses.csv')

# Load habits data for clustering
def load_data():
    """Load habits data from CSV."""
    data = pd.read_csv('habits_data.csv')  # Ensure this CSV contains 'habit_name' and 'time_needed'
    return data

def preprocess_and_cluster(data):
    """Standardize and cluster habits based on time needed."""
    # Standardize the "time_needed" column
    scaler = StandardScaler()
    data['time_needed_scaled'] = scaler.fit_transform(data[['time_needed']])

    # Apply K-means clustering
    kmeans = KMeans(n_clusters=3, random_state=42)
    data['cluster'] = kmeans.fit_predict(data[['time_needed_scaled']])
    
    # Sort clusters based on average time, from shortest to longest
    cluster_order = data.groupby('cluster')['time_needed'].mean().sort_values().index
    cluster_mapping = {cluster: i for i, cluster in enumerate(cluster_order)}
    data['ordered_cluster'] = data['cluster'].map(cluster_mapping)
    
    # Drop the scaled column to simplify output
    data.drop(columns='time_needed_scaled', inplace=True)
    
    return data

def recommend_habits(data, available_time):
    """Recommend habits based on available time."""
    recommendations = []
    for cluster in data['ordered_cluster'].unique():
        cluster_data = data[data['ordered_cluster'] == cluster]
        
        total_time = 0
        selected_habits = []
        
        # Select habits until reaching the user's available time
        for _, row in cluster_data.iterrows():
            if total_time + row['time_needed'] > available_time:
                break
            selected_habits.append(row['habit_name'])
            total_time += row['time_needed']
        
        if selected_habits:
            recommendations.append({
                "cluster": cluster,
                "habits": selected_habits,
                "total_time": total_time
            })
    
    return recommendations

# Endpoints

@app.route('/chatbot', methods=['POST'])
def chatbot_response():
    """Chatbot response based on user input sentiment."""
    data = request.json
    user_input = data.get('input', '')
    score = sentiment_analyzer.polarity_scores(user_input)
    sentiment_score = score['compound']
    
    if sentiment_score >= 0.05:
        detected_sentiment = 'positive'
        response = random.choice(positive_responses)
    elif sentiment_score <= -0.05:
        detected_sentiment = 'negative'
        response = random.choice(negative_responses)
    else:
        detected_sentiment = 'neutral'
        response = random.choice(neutral_responses)

    logging.info(f"Chatbot detected sentiment: {detected_sentiment}")
    return jsonify({"response": response})

@app.route('/ai_writing_therapist', methods=['POST'])
def ai_writing_therapist():
    """Provide writing prompts based on user mood."""
    data = request.json
    mood = data.get('mood', '').lower()
    topics = {
        "good": ["Describe a recent accomplishment you're proud of."],
        "neutral": ["What is a daily routine you enjoy?"],
        "bad": ["Write about a challenge you're currently facing."]
    }

    if mood not in topics:
        return jsonify({"error": "Invalid mood. Please choose from good, neutral, or bad."}), 400

    topic = random.choice(topics[mood])
    return jsonify({"prompt": topic})

@app.route('/recommend_habit', methods=['POST'])
def recommend_habit_endpoint():
    """Recommend a habit based on user data."""
    user_data = request.json
    required_keys = ['exercise_frequency', 'social_media_hours', 'stress_level', 'mindfulness_frequency']
    
    # Validate input
    if not all(key in user_data for key in required_keys):
        return jsonify({'error': 'Invalid input. All fields are required.'}), 400
    
    # Get user input
    exercise_freq = user_data['exercise_frequency']
    social_media_hours = user_data['social_media_hours']
    stress_level = user_data['stress_level']
    mindfulness_freq = user_data['mindfulness_frequency']

    # Recommend a habit based on user input
    try:
        user_input = pd.DataFrame({
            'exercise_frequency': [le_exercise.transform([exercise_freq])[0]],
            'social_media_hours': [le_social_media.transform([social_media_hours])[0]],
            'stress_level': [le_stress.transform([stress_level])[0]],
            'mindfulness_frequency': [le_mindfulness.transform([mindfulness_freq])[0]]
        })

        recommended_habit = model.predict(user_input)
        logging.info(f"Recommended habit: {recommended_habit[0]}")
        return jsonify({'recommended_habit': recommended_habit[0]})
    except ValueError as e:
        logging.error(f"ValueError in recommend_habit: {str(e)}")
        return jsonify({"error": f"Invalid input value: {str(e)}"}), 400

@app.route('/music_recommendation', methods=['POST'])
def music_recommendation():
    """Recommend music based on mood."""
    mood = request.json.get('mood', '').lower()
    filtered_data = music_data[music_data['mood'].str.lower() == mood]

    if filtered_data.empty:
        return jsonify({"error": "No songs found for that mood."}), 404

    recommendations = filtered_data[['title', 'file_path']].to_dict(orient='records')
    return jsonify({"recommendations": recommendations})


@app.route('/habit_clustering', methods=['POST'])
def habit_clustering():
    """Cluster habits based on user’s available time."""
    user_data = request.json
    time_available = user_data.get('time_available', 0)

    # Load and preprocess the data
    habits_data = load_data()
    clustered_data = preprocess_and_cluster(habits_data)

    # Get recommendations based on available time
    recommendations = recommend_habits(clustered_data, time_available)

    # Convert recommendations to native Python types if necessary
    recommendations = convert_to_native(recommendations)

    if not recommendations:
        return jsonify({"error": "No habits fit within your available time."}), 404

    return jsonify({"recommendations": recommendations})

def convert_to_native(data):
    """Convert numpy and pandas data types to native Python types."""
    if isinstance(data, (np.ndarray, pd.Series)):
        return data.tolist()  # Convert to list
    elif isinstance(data, dict):
        # Recursively convert dict values
        return {key: convert_to_native(value) for key, value in data.items()}
    elif isinstance(data, (np.integer, np.float64)):  # Use np.float64 for numpy float
        return data.item()  # Convert numpy scalar to native type
    elif isinstance(data, list):
        return [convert_to_native(item) for item in data]  # Convert items in a list
    return data  # Return data as-is if no conversion is needed

# Start the app
if __name__ == '__main__':
    app.run(debug=True)
