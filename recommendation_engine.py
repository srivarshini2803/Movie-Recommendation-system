import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import TruncatedSVD
from surprise import SVD, Dataset, Reader, KNNBasic
from surprise.model_selection import train_test_split
import re
import json
import csv
from collections import Counter

def load_and_clean_data(filepath='movies_data.csv'):
    df = pd.read_csv(filepath)
    
    # Create a copy and modify it properly
    df = df.copy()
    
    # Fix the fillna operations
    df['Meta_score'] = df['Meta_score'].fillna(df['Meta_score'].mean())
    df['Gross'] = df['Gross'].fillna(0)
    
    # Normalize runtime (extract minutes)
    df['Runtime'] = df['Runtime'].apply(lambda x: int(re.findall(r'\d+', str(x))[0]) if re.findall(r'\d+', str(x)) else 0)
    
    # Convert Gross to numeric (remove $ and commas)
    df['Gross'] = df['Gross'].apply(lambda x: int(re.sub(r'[^\d.]', '', str(x))) if re.sub(r'[^\d.]', '', str(x)) else 0)
    
    # Create a combined text field for content-based filtering
    df['combined_features'] = df['Overview'] + ' ' + df['Genre'] + ' ' + df['Director'] + ' ' + df['Star1'] + ' ' + df['Star2'] + ' ' + df['Star3'] + ' ' + df['Star4']
    
    return df

# Create vectorization and similarity matrices
def create_matrices(df):
    # TF-IDF for Overview
    tfidf_vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
    tfidf_matrix = tfidf_vectorizer.fit_transform(df['Overview'].fillna(''))
    
    # CountVectorizer for Genre and Cast
    count_vectorizer = CountVectorizer(max_features=2000, stop_words='english')
    count_matrix = count_vectorizer.fit_transform(df['combined_features'].fillna(''))
    
    # Compute similarity matrices
    tfidf_sim_matrix = cosine_similarity(tfidf_matrix, tfidf_matrix)
    count_sim_matrix = cosine_similarity(count_matrix, count_matrix)
    
    # Combined similarity (weighted)
    combined_sim = 0.7 * count_sim_matrix + 0.3 * tfidf_sim_matrix
    
    return {
        'tfidf_vectorizer': tfidf_vectorizer,
        'count_vectorizer': count_vectorizer,
        'tfidf_matrix': tfidf_matrix,
        'count_matrix': count_matrix,
        'tfidf_sim_matrix': tfidf_sim_matrix,
        'count_sim_matrix': count_sim_matrix,
        'combined_sim': combined_sim
    }

# Content-based filtering
def get_similar_movies(df, title, matrices, n=None):
    # Get the index of the movie that matches the title
    idx = df[df['Series_Title'] == title].index[0] if len(df[df['Series_Title'] == title]) > 0 else 0
    
    # Get similarity scores
    sim_scores = list(enumerate(matrices['combined_sim'][idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:n+1]  # Exclude the movie itself
    
    # Get movie indices
    movie_indices = [i[0] for i in sim_scores]
    
    # Create explanations
    recommendations = []
    for i, idx in enumerate(movie_indices):
        movie = df.iloc[idx]
        explanation = f"Recommended because of similar {movie['Genre']} genre and featuring {movie['Star1']}"
        if movie['Director'] == df.iloc[idx]['Director']:
            explanation += f" and directed by the same director ({movie['Director']})"
        
        recommendations.append({
            'title': movie['Series_Title'],
            'poster': movie['Poster_Link'],
            'year': movie['Released_Year'],
            'rating': movie['IMDB_Rating'],
            'similarity_score': sim_scores[i][1],
            'explanation': explanation,
            'genre': movie['Genre'],
            'overview': movie['Overview']
        })
    
    return recommendations

# Popularity-based filtering
def get_popular_movies(df, n=None, min_votes=1000, year_range=None, genre=None):
    filtered_df = df.copy()
    
    # Pre-filtering
    if year_range:
        filtered_df = filtered_df[(filtered_df['Released_Year'] >= year_range[0]) & 
                                (filtered_df['Released_Year'] <= year_range[1])]
    
    if genre:
        filtered_df = filtered_df[filtered_df['Genre'].str.contains(genre, case=False, na=False)]
    
    # Filter by minimum votes
    filtered_df = filtered_df[filtered_df['No_of_Votes'] >= min_votes]
    
    # Calculate a weighted rating
    C = filtered_df['IMDB_Rating'].mean()
    m = min_votes
    filtered_df['weighted_rating'] = (filtered_df['No_of_Votes'] / (filtered_df['No_of_Votes'] + m) * 
                                    filtered_df['IMDB_Rating']) + (m / (filtered_df['No_of_Votes'] + m) * C)
    
    # Sort by weighted rating
    popular_movies = filtered_df.sort_values('weighted_rating', ascending=False).head(n)
    
    recommendations = []
    for _, movie in popular_movies.iterrows():
        explanation = f"Highly rated ({movie['IMDB_Rating']}/10) with {movie['No_of_Votes']} votes"
        recommendations.append({
            'title': movie['Series_Title'],
            'poster': movie['Poster_Link'],
            'year': movie['Released_Year'],
            'rating': movie['IMDB_Rating'],
            'votes': movie['No_of_Votes'],
            'weighted_rating': float(movie['weighted_rating']),
            'explanation': explanation,
            'genre': movie['Genre'],
            'overview': movie['Overview']
        })
    
    return recommendations

# Generate user-item matrix for collaborative filtering
def create_user_item_matrix(df, user_ratings=None):
    # Create a simple user-item matrix with synthetic users if no real user data
    if user_ratings is None:
        # Create synthetic user ratings based on movie popularity
        n_users = 100
        n_movies = len(df)
        user_item_matrix = np.zeros((n_users, n_movies))
        
        # Assign ratings based on movie popularity (higher rated movies more likely to be rated)
        for i in range(n_users):
            # Each user rates 20-50 random movies
            n_ratings = np.random.randint(20, 50)
            movie_indices = np.random.choice(n_movies, n_ratings, replace=False)
            
            for idx in movie_indices:
                # Rating influenced by IMDB rating but with some noise
                base_rating = df.iloc[idx]['IMDB_Rating'] / 2  # Convert to 0-5 scale
                noise = np.random.normal(0, 0.5)
                rating = max(0.5, min(5, base_rating + noise))
                user_item_matrix[i, idx] = rating
    else:
        # Use provided user ratings
        user_item_matrix = user_ratings
    
    return user_item_matrix

# Collaborative filtering with KNN
def get_knn_recommendations(df, user_id, user_item_matrix, n=None):
    # Create a KNN model
    model = NearestNeighbors(metric='cosine', algorithm='brute')
    model.fit(user_item_matrix)
    
    # Find similar users
    distances, indices = model.kneighbors([user_item_matrix[user_id]], n_neighbors=5)
    
    # Get movies rated highly by similar users but not seen by the target user
    seen_movies = set(np.where(user_item_matrix[user_id] > 0)[0])
    recommendations = []
    
    for idx in indices.flatten():
        if idx == user_id:
            continue
            
        # Find movies rated highly by this similar user
        similar_user_rated = np.where(user_item_matrix[idx] >= 4)[0]
        for movie_idx in similar_user_rated:
            if movie_idx not in seen_movies and len(recommendations) < n:
                movie = df.iloc[movie_idx]
                similarity = 1 - distances.flatten()[list(indices.flatten()).index(idx)]
                explanation = f"Recommended because users with similar taste rated this movie highly"
                
                recommendations.append({
                    'title': movie['Series_Title'],
                    'poster': movie['Poster_Link'],
                    'year': movie['Released_Year'],
                    'rating': movie['IMDB_Rating'],
                    'similarity_score': float(similarity),
                    'explanation': explanation,
                    'genre': movie['Genre'],
                    'overview': movie['Overview']
                })
                
                seen_movies.add(movie_idx)
    
    return recommendations[:n]

# Collaborative filtering with SVD
def get_svd_recommendations(df, user_id, user_item_matrix, n=None):
    # Prepare data for SVD
    user_indices, movie_indices = np.where(user_item_matrix > 0)
    ratings = user_item_matrix[user_indices, movie_indices]
    
    # Create a DataFrame for surprise
    ratings_dict = {
        'user': user_indices,
        'item': movie_indices,
        'rating': ratings
    }
    ratings_df = pd.DataFrame(ratings_dict)
    
    # Create a Surprise dataset
    reader = Reader(rating_scale=(0.5, 5))
    data = Dataset.load_from_df(ratings_df, reader)
    
    # Split the data for training
    trainset, _ = train_test_split(data, test_size=0.2)
    
    # Train the SVD model
    model = SVD(n_factors=50)
    model.fit(trainset)
    
    # Find movies not seen by the user
    seen_movies = set(np.where(user_item_matrix[user_id] > 0)[0])
    unseen_movies = set(range(len(df))) - seen_movies
    
    # Predict ratings for unseen movies
    predictions = []
    for movie_idx in unseen_movies:
        est_rating = model.predict(user_id, movie_idx).est
        predictions.append((movie_idx, est_rating))
    
    # Sort by predicted rating
    predictions.sort(key=lambda x: x[1], reverse=True)
    
    # Get top N recommendations
    recommendations = []
    for movie_idx, est_rating in predictions[:n]:
        movie = df.iloc[movie_idx]
        explanation = f"Predicted rating for you: {est_rating:.1f}/5.0 based on your rating patterns"
        
        recommendations.append({
            'title': movie['Series_Title'],
            'poster': movie['Poster_Link'],
            'year': movie['Released_Year'],
            'rating': movie['IMDB_Rating'],
            'predicted_rating': float(est_rating),
            'explanation': explanation,
            'genre': movie['Genre'],
            'overview': movie['Overview']
        })
    
    return recommendations

# Person Correlation for user similarity
def pearson_similarity(user1, user2):
    # Find common rated items
    common_items = np.logical_and(user1 > 0, user2 > 0)
    
    if np.sum(common_items) == 0:
        return 0
    
    # Calculate Pearson correlation
    user1_common = user1[common_items]
    user2_common = user2[common_items]
    
    user1_mean = np.mean(user1_common)
    user2_mean = np.mean(user2_common)
    
    numerator = np.sum((user1_common - user1_mean) * (user2_common - user2_mean))
    denominator = np.sqrt(np.sum((user1_common - user1_mean) ** 2) * np.sum((user2_common - user2_mean) ** 2))
    
    if denominator == 0:
        return 0
    
    return numerator / denominator

# Collaborative filtering with Pearson correlation
def get_pearson_recommendations(df, user_id, user_item_matrix, n=None):
    # Calculate similarity with all users
    similarities = []
    for i in range(len(user_item_matrix)):
        if i != user_id:
            similarity = pearson_similarity(user_item_matrix[user_id], user_item_matrix[i])
            similarities.append((i, similarity))
    
    # Sort by similarity
    similarities.sort(key=lambda x: x[1], reverse=True)
    
    # Get top similar users
    top_similar_users = similarities[:5]
    
    # Find movies rated highly by similar users but not seen by target user
    seen_movies = set(np.where(user_item_matrix[user_id] > 0)[0])
    recommendations = []
    
    for similar_user, similarity in top_similar_users:
        if similarity <= 0:
            continue
            
        # Get movies rated highly by this similar user
        rated_movies = np.where(user_item_matrix[similar_user] >= 4)[0]
        
        for movie_idx in rated_movies:
            if movie_idx not in seen_movies and len(recommendations) < n:
                movie = df.iloc[movie_idx]
                explanation = f"Recommended because users with {similarity:.2f} similarity to you rated this highly"
                
                recommendations.append({
                    'title': movie['Series_Title'],
                    'poster': movie['Poster_Link'],
                    'year': movie['Released_Year'],
                    'rating': movie['IMDB_Rating'],
                    'similarity_score': float(similarity),
                    'explanation': explanation,
                    'genre': movie['Genre'],
                    'overview': movie['Overview']
                })
                
                seen_movies.add(movie_idx)
    
    return recommendations[:n]

# Demographic filtering (for cold-start)
def get_demographic_recommendations(df, certificate=None, genres=None, n=20, user_id=None, filters=None):
    filtered_df = df.copy()
    
    # Apply direct parameters first
    if certificate:
        filtered_df = filtered_df[filtered_df['Certificate'] == certificate]
        
    if genres:
        # Handle multiple genres in the list
        genre_filter = '|'.join([g.strip() for g in genres])
        filtered_df = filtered_df[filtered_df['Genre'].str.contains(genre_filter, case=False, na=False)]
    
    # Apply additional filters if provided
    if filters:
        if 'genre' in filters and filters['genre']:
            filtered_df = filtered_df[filtered_df['Genre'].str.contains(filters['genre'], case=False, na=False)]
        
        if 'year_min' in filters and filters['year_min']:
            filtered_df = filtered_df[filtered_df['Released_Year'] >= filters['year_min']]
            
        if 'year_max' in filters and filters['year_max']:
            filtered_df = filtered_df[filtered_df['Released_Year'] <= filters['year_max']]
            
        if 'rating_min' in filters and filters['rating_min']:
            filtered_df = filtered_df[filtered_df['IMDB_Rating'] >= filters['rating_min']]
            
        if 'certificate' in filters and filters['certificate']:
            filtered_df = filtered_df[filtered_df['Certificate'] == filters['certificate']]
    
    # Sort by IMDB rating and number of votes
    filtered_df['score'] = filtered_df['IMDB_Rating'] * np.log1p(filtered_df['No_of_Votes'])
    recommendations_df = filtered_df.sort_values('score', ascending=False).head(n)
    
    recommendations = []
    for _, movie in recommendations_df.iterrows():
        explanation = f"Top rated {movie['Genre']} movie with {movie['IMDB_Rating']}/10 on IMDB"
        
        recommendations.append({
            'title': movie['Series_Title'],
            'poster': movie['Poster_Link'],
            'year': movie['Released_Year'],
            'rating': movie['IMDB_Rating'],
            'votes': movie['No_of_Votes'],
            'explanation': explanation,
            'genre': movie['Genre'],
            'overview': movie['Overview']
        })
    
    return recommendations

# Hybrid recommendations
def get_hybrid_recommendations(df, title=None, user_id=None, matrices=None, user_item_matrix=None, n=None, weights=None):
    if weights is None:
        weights = {
            'content': 0.3,
            'collaborative': 0.4,
            'popularity': 0.3
        }
    
    all_recommendations = {}
    
    # Get content-based recommendations if title is provided
    if title and matrices:
        content_recs = get_similar_movies(df, title, matrices, n=n)
        for rec in content_recs:
            all_recommendations[rec['title']] = {
                'score': weights['content'] * rec['similarity_score'],
                'data': rec,
                'methods': ['content']
            }
    
    # Get collaborative recommendations if user_id is provided
    if user_id is not None and user_item_matrix is not None:
        collab_recs = get_svd_recommendations(df, user_id, user_item_matrix, n=n)
        for rec in collab_recs:
            if rec['title'] in all_recommendations:
                all_recommendations[rec['title']]['score'] += weights['collaborative'] * (rec['predicted_rating'] / 5)
                all_recommendations[rec['title']]['methods'].append('collaborative')
            else:
                all_recommendations[rec['title']] = {
                    'score': weights['collaborative'] * (rec['predicted_rating'] / 5),
                    'data': rec,
                    'methods': ['collaborative']
                }
    
    # Get popularity-based recommendations
    popular_recs = get_popular_movies(df, n=n)
    for rec in popular_recs:
        if rec['title'] in all_recommendations:
            all_recommendations[rec['title']]['score'] += weights['popularity'] * (rec['weighted_rating'] / 10)
            all_recommendations[rec['title']]['methods'].append('popularity')
        else:
            all_recommendations[rec['title']] = {
                'score': weights['popularity'] * (rec['weighted_rating'] / 10),
                'data': rec,
                'methods': ['popularity']
            }
    
    # Sort by combined score
    sorted_recs = sorted(all_recommendations.values(), key=lambda x: x['score'], reverse=True)[:n]
    
    # Format recommendations with explanations
    final_recommendations = []
    for rec in sorted_recs:
        explanation = "Recommended based on "
        if 'content' in rec['methods'] and 'collaborative' in rec['methods']:
            explanation += "both movie content similarity and user preferences"
        elif 'content' in rec['methods']:
            explanation += "content similarity to movies you like"
        elif 'collaborative' in rec['methods']:
            explanation += "preferences of users with similar taste"
        elif 'popularity' in rec['methods']:
            explanation += "overall popularity and high ratings"
        
        rec_data = rec['data'].copy()
        rec_data['hybrid_score'] = rec['score']
        rec_data['recommendation_methods'] = rec['methods']
        rec_data['explanation'] = explanation
        
        final_recommendations.append(rec_data)
    
    return final_recommendations

# Personalized recommendations based on user history
def get_personalized_recommendations(df, user_history, matrices, n=None):
    # Extract genres from user history
    genres = []
    directors = []
    actors = []
    
    for title in user_history:
        if title in df['Series_Title'].values:
            movie = df[df['Series_Title'] == title].iloc[0]
            genres.extend(movie['Genre'].split(', '))
            directors.append(movie['Director'])
            actors.extend([movie['Star1'], movie['Star2'], movie['Star3'], movie['Star4']])
    
    # Count frequencies
    genre_counter = Counter(genres)
    director_counter = Counter(directors)
    actor_counter = Counter(actors)
    
    # Get top preferences
    top_genres = [genre for genre, _ in genre_counter.most_common(3)]
    top_directors = [director for director, _ in director_counter.most_common(2)]
    top_actors = [actor for actor, _ in actor_counter.most_common(3)]
    
    # Filter movies matching user preferences
    filtered_df = df.copy()
    
    # Exclude already watched movies
    filtered_df = filtered_df[~filtered_df['Series_Title'].isin(user_history)]
    
    # Score movies based on preference matching
    filtered_df['pref_score'] = 0
    
    for genre in top_genres:
        filtered_df['pref_score'] += filtered_df['Genre'].str.contains(genre, case=False, na=False).astype(int) * 2
    
    for director in top_directors:
        filtered_df['pref_score'] += (filtered_df['Director'] == director).astype(int) * 3
    
    for actor in top_actors:
        actor_columns = ['Star1', 'Star2', 'Star3', 'Star4']
        for col in actor_columns:
            filtered_df['pref_score'] += (filtered_df[col] == actor).astype(int) * 1
    
    # Combine with rating for final score
    filtered_df['final_score'] = filtered_df['pref_score'] * filtered_df['IMDB_Rating']
    
    # Get top recommendations
    recommendations_df = filtered_df.sort_values('final_score', ascending=False).head(n)
    
    recommendations = []
    for _, movie in recommendations_df.iterrows():
        explanation = "Recommended based on your interest in "
        matching_genres = [g for g in top_genres if g in movie['Genre']]
        matching_directors = [d for d in top_directors if d == movie['Director']]
        matching_actors = [a for a in top_actors if a in [movie['Star1'], movie['Star2'], movie['Star3'], movie['Star4']]]
        
        explanation_parts = []
        if matching_genres:
            explanation_parts.append(f"{', '.join(matching_genres)} movies")
        if matching_directors:
            explanation_parts.append(f"director {', '.join(matching_directors)}")
        if matching_actors:
            explanation_parts.append(f"actors {', '.join(matching_actors)}")
        
        explanation += " and ".join(explanation_parts)
        
        recommendations.append({
            'title': movie['Series_Title'],
            'poster': movie['Poster_Link'],
            'year': movie['Released_Year'],
            'rating': movie['IMDB_Rating'],
            'preference_score': float(movie['pref_score']),
            'explanation': explanation,
            'genre': movie['Genre'],
            'overview': movie['Overview']
        })
    
    return recommendations

# Pre-filtering and post-filtering
def filter_recommendations(recommendations, filters=None):
    if not filters:
        return recommendations
    
    filtered_recs = recommendations.copy()
    
    # Apply filters
    if 'genre' in filters and filters['genre']:
        filtered_recs = [rec for rec in filtered_recs if filters['genre'].lower() in rec['genre'].lower()]
    
    if 'year_min' in filters and filters['year_min']:
        filtered_recs = [rec for rec in filtered_recs if rec['year'] >= filters['year_min']]
    
    if 'year_max' in filters and filters['year_max']:
        filtered_recs = [rec for rec in filtered_recs if rec['year'] <= filters['year_max']]
    
    if 'rating_min' in filters and filters['rating_min']:
        filtered_recs = [rec for rec in filtered_recs if rec['rating'] >= filters['rating_min']]
    
    return filtered_recs

# Export recommendations to JSON or CSV
def export_recommendations(recommendations, format='json', filename='recommendations'):
    if format.lower() == 'json':
        with open(f"{filename}.json", 'w', encoding='utf-8') as f:
            json.dump(recommendations, f, indent=4)
        return f"{filename}.json"
    
    elif format.lower() == 'csv':
        # Get all unique keys
        keys = set()
        for rec in recommendations:
            keys.update(rec.keys())
        
        with open(f"{filename}.csv", 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=list(keys))
            writer.writeheader()
            writer.writerows(recommendations)
        
        return f"{filename}.csv"
    
    return None

# Initialize everything
def initialize_system(filepath='movies_data.csv'):
    df = load_and_clean_data(filepath)
    matrices = create_matrices(df)
    user_item_matrix = create_user_item_matrix(df)
    
    return {
        'df': df,
        'matrices': matrices,
        'user_item_matrix': user_item_matrix
    }