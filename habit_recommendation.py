import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder

# Load habit recommendation data
data = pd.read_csv('habit_data.csv')

# Preprocess the data
le_exercise = LabelEncoder()
le_social_media = LabelEncoder()
le_stress = LabelEncoder()
le_mindfulness = LabelEncoder()

data['exercise_frequency'] = le_exercise.fit_transform(data['exercise_frequency'])
data['social_media_hours'] = le_social_media.fit_transform(data['social_media_hours'])
data['stress_level'] = le_stress.fit_transform(data['stress_level'])
data['mindfulness_frequency'] = le_mindfulness.fit_transform(data['mindfulness_frequency'])

# Features and target variable
X = data.drop('recommended_habit', axis=1)
y = data['recommended_habit']

# Train the Decision Tree model
clf = DecisionTreeClassifier(random_state=42)
clf.fit(X, y)

def recommend_habit(exercise_freq, social_media_hours, stress_level, mindfulness_freq):
    # Transform user input into the same format as training data
    try:
        user_input = pd.DataFrame({
            'exercise_frequency': [le_exercise.transform([exercise_freq])[0]],
            'social_media_hours': [le_social_media.transform([social_media_hours])[0]],
            'stress_level': [le_stress.transform([stress_level])[0]],
            'mindfulness_frequency': [le_mindfulness.transform([mindfulness_freq])[0]]
        })

        # Predict the habit
        recommended_habit = clf.predict(user_input)
        return recommended_habit[0]
    except ValueError as e:
        return f"Error: {str(e)}"

# Simple questionnaire
def main():
    print("Please answer the following questions to get a habit recommendation.")

    exercise_frequency = input("How often do you exercise? (Daily/A few times a week/Rarely/Never): ")
    social_media_hours = input("How many hours do you spend on social media? (Less than 1 hour/1-2 hours/2-3 hours/More than 3 hours): ")
    stress_level = input("How would you rate your stress level? (Low/Moderate/High): ")
    mindfulness_frequency = input("How often do you practice mindfulness? (Daily/A few times a week/Rarely/Never): ")

    habit = recommend_habit(exercise_frequency, social_media_hours, stress_level, mindfulness_frequency)
    print(f"Recommended Habit: {habit}")

if __name__ == "__main__":
    main()
