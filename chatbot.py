# chatbot.py
import logging
import random
import csv
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# Configure logging
logging.basicConfig(filename='app.log', level=logging.INFO, 
                    format='%(asctime)s %(levelname)s: %(message)s')

# Initialize the sentiment analyzer
sentiment_analyzer = SentimentIntensityAnalyzer()

# Response pools for different sentiments
positive_responses = []
neutral_responses = []
negative_responses = []

# Function to load responses from the CSV file
def load_responses(filename):
    with open(filename, mode='r', encoding='utf-8') as file:
        reader = csv.DictReader(file)
        for row in reader:
            if row['Sentiment'] == 'Positive':
                positive_responses.append(row['Response'])
            elif row['Sentiment'] == 'Neutral':
                neutral_responses.append(row['Response'])
            elif row['Sentiment'] == 'Negative':
                negative_responses.append(row['Response'])

# Load responses from CSV (Make sure 'responses.csv' is present)
load_responses('responses.csv')

# Function to analyze sentiment
def analyze_sentiment(text):
    score = sentiment_analyzer.polarity_scores(text)
    return score['compound']

# Function to generate a response based on sentiment
def generate_response(user_input):
    sentiment_score = analyze_sentiment(user_input)
    
    if sentiment_score >= 0.05:
        detected_sentiment = 'positive'
        response = random.choice(positive_responses)
    elif sentiment_score <= -0.05:
        detected_sentiment = 'negative'
        response = random.choice(negative_responses)
    else:
        detected_sentiment = 'neutral'
        response = random.choice(neutral_responses)
    
    logging.info(f"Detected Sentiment: {detected_sentiment}")
    return response

# New function to handle incoming requests
def handle_chatbot_request(user_input):
    return generate_response(user_input)

if __name__ == "__main__":
    # For testing individually with conversation loop
    print("Chatbot is ready! Type 'exit' to end the conversation.")
    while True:
        user_input = input("You: ")
        if user_input.lower() == 'exit':
            print("Ending conversation. Goodbye!")
            break
        response = handle_chatbot_request(user_input)
        print(f"Chatbot: {response}")