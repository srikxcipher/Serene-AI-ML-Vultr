# SERENE-AI: AI-Driven Wellness Assistant

## Overview

**SERENE-AI** is an advanced AI-driven personal wellness assistant designed to support users in managing their well-being through data-driven insights. It combines state-of-the-art machine learning models and APIs to offer personalized recommendations for habits, music, writing prompts, and more. Whether you want to develop healthier routines, receive mood-based music suggestions, or engage in therapeutic writing, **SERENE-AI** can assist you in your wellness journey.

The project integrates various machine learning models for personalized habit recommendations, sentiment analysis, mood-based music recommendations, and therapeutic writing prompts. All of these capabilities are hosted and available through APIs, which can be easily consumed by Android apps, web apps, or any other frontend application.

---

## Features

- **Personalized Habit Recommendations**: Based on users' input data (e.g., exercise frequency, stress level, mindfulness practice), SERENE-AI recommends tailored wellness habits.
- **AI Writing Therapist**: Provides personalized writing prompts based on user mood (e.g., "Describe a recent accomplishment" for a good mood, or "Write about a challenge you're facing" for a bad mood).
- **Sentiment Analysis-Based Chatbot**: An intelligent chatbot that responds based on the sentiment of the user's input (positive, neutral, or negative).
- **Mood-Based Music Recommendations**: Provides music suggestions based on the user's mood, helping improve emotional well-being.
- **Habit Clustering**: Recommends habits based on available time by clustering habits into different categories (e.g., quick habits, moderate habits, time-consuming habits).

---

## Tech Stack

- **Backend Framework**: Flask (Python)
- **Machine Learning**: 
  - Decision Trees for personalized habit recommendations
  - Sentiment Analysis (VADER) for chatbot responses
  - KMeans Clustering for habit clustering
- **Data Storage**: CSV files for habit data and music data
- **API**: RESTful API for easy integration with frontend apps (Android, Web)
- **Deployment**: Vultr (for hosting the models and APIs)

---

## Requirements

### Software Requirements:

- Python 3.x
- Flask
- Flask-CORS
- scikit-learn
- pandas
- vaderSentiment
- numpy
- csv
- random
- logging
- json

### Python Libraries to Install:

```bash
pip install flask flask-cors scikit-learn pandas vaderSentiment numpy
```

Setup Instructions
------------------

Follow these steps to set up **SERENE-AI** on your local machine or server:

### 1\. Clone the Repository

Start by cloning the repository to your local machine or cloud server.

```bash
git clone https://github.com/srikxcipher/SERENE-AI.git
cd SERENE-AI
```
### 2\. Install Required Python Packages

Create and activate a virtual environment for the project (recommended):

```bash
python -m venv venv
source venv/bin/activate  # On Windows use 'venv\Scripts\activate'`
```
Then, install all the required Python libraries:

```bash
pip install -r requirements.txt
```

### 3\. Prepare the Data

**Habit Data**: Make sure the `habit_data.csv` file is present in the project directory. This file should contain historical user data related to habits, including columns like:

-   `exercise_frequency`
-   `social_media_hours`
-   `stress_level`
-   `mindfulness_frequency`
-   `recommended_habit`

**Music Data**: Make sure the `music_data.csv` file is present in the project directory. This file should contain columns like:

-   `mood`
-   `title`
-   `file_path`

**Response Data**: Ensure the `responses.csv` file is available to store chatbot responses with sentiment classifications (positive, neutral, negative).

### 4\. Launch the Flask App

To start the application, run the following command:

```bash
python app.py
```
This will start the Flask development server on `http://127.0.0.1:5000/` (or the configured host/port). The application will be running and accessible for API requests.

* * * * *

API Endpoints
-------------

### 1\. **/chatbot (POST)**

The chatbot responds based on the sentiment of the user's input.

#### Request Example:

```json
{
  "input": "I feel great today!"
}
```
#### Response Example:

```json
{
  "response": "That's awesome! Keep it up!"
}
```
### 2\. **/ai_writing_therapist (POST)**

This endpoint provides a writing prompt based on the user's mood.

#### Request Example:

```json
{
  "mood": "good"
}
```
#### Response Example:

```json
{
  "prompt": "Describe a recent accomplishment you're proud of."
}
```
### 3\. **/recommend_habit (POST)**

Recommends a personalized habit based on user input (e.g., exercise frequency, social media hours, stress level, mindfulness frequency).

#### Request Example:

```json
{
  "exercise_frequency": "Daily",
  "social_media_hours": "1",
  "stress_level": "High",
  "mindfulness_frequency": "Weekly"
}
```
#### Response Example:

```json
{
  "recommended_habit": "Meditation"
}
```
### 4\. **/music_recommendation (POST)**

Recommends music based on the user's mood.

#### Request Example:

```json
{
  "mood": "happy"
}
```
#### Response Example:
```json
{
  "recommendations": [
    {
      "title": "Feel Good Inc.",
      "file_path": "/music/feel_good_inc.mp3"
    }
  ]
}
```
### 5\. **/habit_clustering (POST)**

Recommends habits based on available time. Habits are clustered by time requirements (e.g., short, medium, or long duration).

#### Request Example:

```json
{
  "time_available": 30
}
```
#### Response Example:
```json
{
  "recommendations": [
    {
      "cluster": 0,
      "habits": ["Stretching", "Breathing Exercises"],
      "total_time": 25
    }
  ]
}
```
* * * * *

Usage
-----

### 1\. **Integrating with Android or Web Apps**

The APIs are designed to be consumed by any client application that can make HTTP requests. In your Android app (or other frontend), you can use the following approach:

1.  **POST Requests**: Make POST requests to the endpoints and pass the required data (as shown in the examples).
2.  **Parse the JSON Response**: After making the request, parse the JSON response to display the recommendations, writing prompts, or chatbot responses to users.
3.  **User Interaction**: Use the chatbot and AI writing therapist to engage users, and provide personalized wellness content such as music and habit suggestions based on user preferences and moods.

* * * * *

Deployment
----------

**SERENE-AI** is hosted on **Vultr** for scalability and reliability.
