# habit_clustering.py
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

def load_data():
    # Load the habits data
    data = pd.read_csv('habits_data.csv')  # File with columns 'habit_name' and 'time_needed'
    return data

def preprocess_and_cluster(data):
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
    # Calculate possible time-ordered clusters based on user’s available time
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

def main():
    # Load data and preprocess
    data = load_data()
    clustered_data = preprocess_and_cluster(data)
    
    # User input for available time
    available_time = int(input("Enter the time you have available (in minutes): "))
    recommendations = recommend_habits(clustered_data, available_time)
    
    # Output results
    if recommendations:
        print("\nRecommended habits based on your available time:")
        for rec in recommendations:
            print(f"Cluster {rec['cluster']} (Total Time: {rec['total_time']} mins):")
            for habit in rec['habits']:
                print(f"- {habit}")
    else:
        print("No habits fit within your available time.")

if __name__ == '__main__':
    main()
