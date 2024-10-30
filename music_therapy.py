# music_therapy.py
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import pygame

# Initialize pygame mixer
pygame.mixer.init()

# Load the dataset (Ensure CSV has columns: 'title', 'file_path', 'mood', and 'feature1', 'feature2', etc.)
data = pd.read_csv('music_data.csv')

# Function to get music recommendations based on mood
def get_recommendations(mood):
    filtered_data = data[data['mood'].str.lower() == mood.lower()]
    
    if filtered_data.empty:
        return None

    # Calculate cosine similarity for music features
    features = filtered_data[['feature1', 'feature2', 'feature3']]
    sim_scores = cosine_similarity(features).mean(axis=1)
    top_indices = sim_scores.argsort()[-5:][::-1]  # Get top 5 recommendations

    return filtered_data.iloc[top_indices]

# Function to play the song with option to stop
def play_song(file_path):
    pygame.mixer.music.load(file_path)
    pygame.mixer.music.play()

    # Wait for user input to stop music
    input("Press Enter to stop the music.\n")
    pygame.mixer.music.stop()  # Stop the music when Enter is pressed

# Main function
def main():
    print("Welcome to the Music Therapy!")
    while True:
        mood = input("Enter your mood (fun/relaxation/motivation): ").strip().lower()
        recommendations = get_recommendations(mood)

        if recommendations is not None:
            print("Here are some recommendations:")
            for index, row in recommendations.iterrows():
                print(f"- {row['title']}")
                play_song(row['file_path'])
                break  # Stop after one song and prompt for mood question
            print("Music stopped.")
        else:
            print("No songs found for that mood.")

        # Ask if the user wants to try another mood
        if input("Try another mood? (yes/no): ").strip().lower() != 'yes':
            break

if __name__ == "__main__":
    main()
