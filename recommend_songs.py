import pandas as pd
import sys
import json
import os

def recommend_songs(liked_songs, matrix_file='combined_similarity_matrix.csv', top_n=5):
    """
    Recommends songs based on a list of liked songs using a similarity matrix.

    Args:
        liked_songs (list): List of song titles the user likes.
        matrix_file (str): Path to the CSV file containing the similarity matrix.
        top_n (int): Number of recommendations to return.

    Returns:
        list: A list of recommended song titles.
    """
    try:
        # Load the similarity matrix
        # Assuming the first column is 'song_title' and it should be the index
        if not os.path.exists(matrix_file):
            return {"error": f"File not found: {matrix_file}"}
            
        df = pd.read_csv(matrix_file, index_col=0)
        
        # Filter out liked songs that are not in our database
        valid_liked_songs = [song for song in liked_songs if song in df.index]
        
        if not valid_liked_songs:
            return [] # No valid songs to base recommendations on

        # Sum the similarity scores for the liked songs
        # We select the columns corresponding to the liked songs and sum them across the rows
        # This gives us a total similarity score for each song in the database relative to the user's likes
        
        # Note: The matrix is symmetric, so columns and rows are the same.
        # We want to find rows (songs) that are similar to the columns (liked songs).
        
        # Create a series to hold the aggregated scores
        # Initialize with zeros
        aggregated_scores = pd.Series(0, index=df.index)
        
        for song in valid_liked_songs:
            aggregated_scores += df[song]

        # Remove the songs the user has already liked from the recommendations
        aggregated_scores = aggregated_scores.drop(valid_liked_songs, errors='ignore')

        # Sort by score in descending order and take the top N
        recommendations = aggregated_scores.sort_values(ascending=False).head(top_n)
        
        return recommendations.index.tolist()

    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    # Expecting a JSON string of liked songs as the first argument
    # Example: python recommend_songs.py '["Song A", "Song B"]'
    
    if len(sys.argv) > 1:
        try:
            input_data = " ".join(sys.argv[1:])
            liked_songs = json.loads(input_data)
            
            # Ensure it's a list
            if not isinstance(liked_songs, list):
                print(json.dumps({"error": "Input must be a JSON list of song titles."}))
                sys.exit(1)
                
            # Get recommendations
            recs = recommend_songs(liked_songs)
            
            # Output as JSON
            print(json.dumps(recs))
            
        except json.JSONDecodeError:
            print(json.dumps({"error": "Invalid JSON input."}))
            sys.exit(1)
        except Exception as e:
             print(json.dumps({"error": f"An unexpected error occurred: {str(e)}"}))
             sys.exit(1)
    else:
        print(json.dumps({"error": "No input provided. Usage: python recommend_songs.py '[\"Song1\", \"Song2\"]'"}))
        sys.exit(1)
