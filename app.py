from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import os
import pandas as pd
import numpy as np
import json
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
from recommendation_engine import (
    initialize_system, get_similar_movies, get_popular_movies,
    get_knn_recommendations, get_svd_recommendations, get_pearson_recommendations,
    get_demographic_recommendations, get_hybrid_recommendations, get_personalized_recommendations,
    filter_recommendations, export_recommendations
)



# Check if data file exists
if not os.path.exists('movies_data.csv'):
    raise FileNotFoundError("movies_data.csv not found. Please ensure the dataset is in the root directory.")
app = Flask(__name__, static_folder='static')

# Update CORS configuration
CORS(app, resources={r"/*": {"origins": "*"}})
# Initialize the system
system = initialize_system('movies_data.csv')
df = system['df']
matrices = system['matrices']
user_item_matrix = system['user_item_matrix']

# Store user interaction history
user_interactions = {}

# Demographic recommendation endpoint
@app.route('/recommend/demographic', methods=['POST'])
def demographic_recommendations():
    try:
        logging.info("=== Starting demographic_recommendations() function ===")
        data = request.get_json()
        logging.info(f"Received request data: {data}")
        
        age = int(data.get('age', 25))
        occupation = data.get('occupation', '')
        mood = data.get('mood', '')
        user_id = data.get('user_id', 'guest')
        
        logging.info(f"Processing parameters - age: {age}, occupation: '{occupation}', mood: '{mood}', user_id: '{user_id}'")
        
        # Map age to certificate
        if age < 13:
            certificate = 'U'
            logging.info(f"Age {age} mapped to certificate: U")
        elif age < 18:
            certificate = 'UA'
            logging.info(f"Age {age} mapped to certificate: UA")
        else:
            certificate = 'A'
            logging.info(f"Age {age} mapped to certificate: A")
            
        # Map occupation to genre preferences
        genre_preferences = {
            'student': 'Comedy, Adventure',
            'professional': 'Drama, Thriller',
            'artist': 'Drama, Romance',
            'engineer': 'Sci-Fi, Action',
            'retired': 'Drama, Biography'
        }
        
        # Map mood to additional genres
        mood_genres = {
            'happy': 'Comedy, Musical',
            'sad': 'Drama, Romance',
            'adventurous': 'Adventure, Action',
            'relaxed': 'Drama, Comedy',
            'thoughtful': 'Drama, Mystery'
        }
        
        # Combine genre preferences
        base_genres = genre_preferences.get(occupation, 'Drama, Comedy').split(',')
        mood_based = mood_genres.get(mood, 'Drama').split(',')
        combined_genres = list(set(base_genres + mood_based))
        logging.info(f"Combined genres: {combined_genres}")
        
        # Get recommendations
        logging.info(f"Getting recommendations with certificate: {certificate}, genres: {combined_genres}")
        recommendations = get_demographic_recommendations(
            df, 
            certificate=certificate, 
            genres=combined_genres,
            user_id=user_id
        )
        logging.info(f"Generated {len(recommendations)} recommendations")
        
        return jsonify({
            'status': 'success',
            'data': recommendations
        })
        
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': f'Error processing demographic data: {str(e)}'
        }), 500

# Helper function to track user interactions
def track_interaction(user_id, movie_title, interaction_type, rating=None):
    if user_id not in user_interactions:
        user_interactions[user_id] = {'viewed': [], 'liked': [], 'rated': {}}
    
    if interaction_type == 'view':
        if movie_title not in user_interactions[user_id]['viewed']:
            user_interactions[user_id]['viewed'].append(movie_title)
    elif interaction_type == 'like':
        if movie_title not in user_interactions[user_id]['liked']:
            user_interactions[user_id]['liked'].append(movie_title)
    elif interaction_type == 'rate' and rating is not None:
        user_interactions[user_id]['rated'][movie_title] = rating

# Home route - serve top rated movies
@app.route('/')
def home():
    top_movies = get_popular_movies(df, n=90)
    return jsonify({
        'status': 'success',
        'data': {
            'top_movies': top_movies
        }
    })

# Text-based search route
@app.route('/search/text')
def text_search():
    try:
        query = request.args.get('query', '').strip()
        if not query:
            return jsonify({
                'status': 'error',
                'message': 'Search query is required'
            }), 400

        if "+" in query:
            query = query.replace("+", " ")

        # Search in multiple fields
        results = df[df['Series_Title'].str.contains(query, case=False, na=False) | 
                    df['Overview'].str.contains(query, case=False, na=False) |
                    df['Director'].str.contains(query, case=False, na=False) |
                    df['Genre'].str.contains(query, case=False, na=False)]

        if len(results) == 0:
            return jsonify({
                'status': 'success',
                'message': 'No movies found matching your search',
                'count': 0,
                'data': []
            })

        # Format results
        formatted_results = [{
            'title': movie['Series_Title'],
            'poster': movie['Poster_Link'],
            'year': movie['Released_Year'],
            'rating': movie['IMDB_Rating'],
            'genre': movie['Genre'],
            'overview': movie['Overview']
        } for _, movie in results.iterrows()]

        return jsonify({
            'status': 'success',
            'count': len(formatted_results),
            'data': formatted_results
        })

    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': f'An error occurred while searching: {str(e)}'
        }), 500

# Filter-based search route
@app.route('/search/filter')
def filter_search():
    try:
        logging.info("=== Starting filter_search() function ===")
        logging.info(f"Request args: {request.args}")
        
        # Get filter parameters with validation
        genre = request.args.get('genre', '').strip()
        certificate = request.args.get('certificate', '').strip()
        logging.info(f"Initial parameters - genre: '{genre}', certificate: '{certificate}'")
        
        # Initialize default values
        year_min = 0
        year_max = 9999
        rating_min = 0
        logging.info(f"Default values - year_min: {year_min}, year_max: {year_max}, rating_min: {rating_min}")
        
        # Validate year_min if provided
        if request.args.get('year_min'):
            logging.info(f"Processing year_min parameter: '{request.args.get('year_min')}'")
            try:
                year_min = int(request.args.get('year_min'))
                logging.info(f"Converted year_min to int: {year_min}")
                if year_min < 1800:  # Reasonable minimum year
                    logging.info(f"year_min {year_min} is below minimum threshold, setting to 1800")
                    year_min = 1800
            except (ValueError, TypeError) as e:
                logging.error(f"Error converting year_min: {e}")
                return jsonify({
                    'status': 'error',
                    'message': 'Year min must be a valid year (e.g., 1800 or later)'
                }), 400

        # Validate year_max if provided
        if request.args.get('year_max'):
            logging.info(f"Processing year_max parameter: '{request.args.get('year_max')}'")
            try:
                year_max = int(request.args.get('year_max'))
                logging.info(f"Converted year_max to int: {year_max}")
                if year_max > 2030:  # Reasonable maximum year
                    logging.info(f"year_max {year_max} is above maximum threshold, setting to 2030")
                    year_max = 2030
            except (ValueError, TypeError) as e:
                logging.error(f"Error converting year_max: {e}")
                return jsonify({
                    'status': 'error',
                    'message': 'Year max must be a valid year'
                }), 400

        # Validate year range
        if year_min > year_max:
            logging.error(f"Invalid year range: year_min ({year_min}) > year_max ({year_max})")
            return jsonify({
                'status': 'error',
                'message': 'Year min cannot be greater than year max'
            }), 400

        # Validate rating_min if provided
        if request.args.get('rating_min'):
            logging.info(f"Processing rating_min parameter: '{request.args.get('rating_min')}'")
            try:
                rating_min = float(request.args.get('rating_min'))
                logging.info(f"Converted rating_min to float: {rating_min}")
                if rating_min < 0:
                    logging.info(f"rating_min {rating_min} is negative, setting to 0")
                    rating_min = 0
                elif rating_min > 10:
                    logging.info(f"rating_min {rating_min} is above maximum threshold, setting to 10")
                    rating_min = 10
            except (ValueError, TypeError) as e:
                logging.error(f"Error converting rating_min: {e}")
                return jsonify({
                    'status': 'error',
                    'message': 'Rating must be a number between 0 and 10'
                }), 400

        # Start with all movies
        results = df.copy()
        logging.info(f"Initial dataframe size: {len(results)} rows")

        # Apply filters
        if genre:
            logging.info(f"Applying genre filter: '{genre}'")
            results = results[results['Genre'].str.contains(genre, case=False, na=False)]
            logging.info(f"After genre filter: {len(results)} rows remaining")
        if certificate:
            logging.info(f"Applying certificate filter: '{certificate}'")
            results = results[results['Certificate'].str.contains(certificate, case=False, na=False)]
            logging.info(f"After certificate filter: {len(results)} rows remaining")
        if year_min > 0:
            logging.info(f"Applying year_min filter: {year_min}")
            # Convert Released_Year to integer before comparison (only once)
            # Convert Released_Year to numeric and drop NaN values before filtering
            results['Released_Year'] = pd.to_numeric(results['Released_Year'], errors='coerce')
            results = results.dropna(subset=['Released_Year'])
            
            # Apply both min and max filters together for better performance
            if year_min > 0 and year_max < 9999:
                results = results[(results['Released_Year'] >= year_min) & (results['Released_Year'] <= year_max)]
                logging.info(f"After year range filter ({year_min}-{year_max}): {len(results)} rows remaining")
            elif year_min > 0:
                results = results[results['Released_Year'] >= year_min]
                logging.info(f"After year_min filter: {len(results)} rows remaining")
            elif year_max < 9999:
                results = results[results['Released_Year'] <= year_max]
                logging.info(f"After year_max filter: {len(results)} rows remaining")
        if rating_min > 0:
            logging.info(f"Applying rating_min filter: {rating_min}")
            # Filter by rating, handling NaN values
            results = results[results['IMDB_Rating'].notna() & (results['IMDB_Rating'] >= rating_min)]
            logging.info(f"After rating_min filter: {len(results)} rows remaining")

        if len(results) == 0:
            logging.info("No results found after applying all filters")
            return jsonify({
                'status': 'success',
                'message': 'No movies found matching your filters',
                'count': 0,
                'data': []
            })

        # Format results
        logging.info(f"Formatting {len(results)} results for response")
        formatted_results = [{
            'title': movie['Series_Title'],
            'poster': movie['Poster_Link'],
            'year': movie['Released_Year'],
            'rating': movie['IMDB_Rating'],
            'genre': movie['Genre'],
            'certificate': movie['Certificate'],
            'overview': movie['Overview']
        } for _, movie in results.iterrows()]

        logging.info(f"Returning {len(formatted_results)} formatted results")
        return jsonify({
            'status': 'success',
            'count': len(formatted_results),
            'data': formatted_results
        })

    except Exception as e:
        logging.error(f"Exception in filter_search: {str(e)}", exc_info=True)
        return jsonify({
            'status': 'error',
            'message': f'An error occurred while filtering: {str(e)}'
        }), 500



@app.route('/movie/<title>')
def movie_detail(title):
    print("@app.route('/movie/<title>')")
    # Track view interaction
    user_id = request.args.get('user_id', 'guest')
    track_interaction(user_id, title, 'view')
    
    # Get movie details
    movie = df[df['Series_Title'].str.lower() == title.lower()]
    if len(movie) == 0:
        # Try partial matching if exact match fails
        movie = df[df['Series_Title'].str.lower().str.contains(title.lower())]
    if len(movie) == 0:
        return jsonify({
            'status': 'error',
            'message': 'Movie not found'
        }), 404
    
    movie = movie.iloc[0]
    
    # Helper function to convert numpy types to Python types
    def convert_to_python_type(value):
        if pd.isna(value):
            return None
        if isinstance(value, (np.int64, np.int32, np.int16, np.int8)):
            return int(value)
        if isinstance(value, (np.float64, np.float32)):
            return float(value)
        return value
    
    # Format details with type conversion for all numeric values
    details = {
        'title': str(movie['Series_Title']),
        'poster': str(movie['Poster_Link']),
        'year': convert_to_python_type(movie['Released_Year']),
        'certificate': str(movie['Certificate']),
        'runtime': str(movie['Runtime']),
        'genre': str(movie['Genre']),
        'rating': convert_to_python_type(movie['IMDB_Rating']),
        'metascore': convert_to_python_type(movie['Meta_score']),
        'overview': str(movie['Overview']),
        'director': str(movie['Director']),
        'stars': [str(movie['Star1']), str(movie['Star2']), str(movie['Star3']), str(movie['Star4'])],
        'votes': convert_to_python_type(movie['No_of_Votes']),
        'gross': str(movie['Gross'])
    }
    
    return jsonify({
        'status': 'success',
        'data': details
    })
# Recommendation routes
@app.route('/recommend')
def hybrid_recommendations():
    title = request.args.get('title', '')
    user_id = request.args.get('user_id', 'guest')
    
    # Get filter parameters
    genre = request.args.get('genre', '')
    year_min = request.args.get('year_min', 0, type=int)
    year_max = request.args.get('year_max', 3000, type=int)
    rating_min = request.args.get('rating_min', 0, type=float)
    
    filters = {}
    if genre:
        filters['genre'] = genre
    if year_min > 0:
        filters['year_min'] = year_min
    if year_max < 3000:
        filters['year_max'] = year_max
    if rating_min > 0:
        filters['rating_min'] = rating_min
    
    # Get hybrid recommendations
    recommendations = get_hybrid_recommendations(
        df, title=title, user_id=int(user_id) if user_id.isdigit() else None,
        matrices=matrices, user_item_matrix=user_item_matrix, n=20
    )
    
    # Apply post-filtering
    if filters:
        recommendations = filter_recommendations(recommendations, filters)
    
    return jsonify({
        'status': 'success',
        'count': len(recommendations),
        'data': recommendations
    })

@app.route('/recommend/content')
def content_recommendations():
    title = request.args.get('title', '')
    
    if not title or title not in df['Series_Title'].values:
        return jsonify({
            'status': 'error',
            'message': 'Valid title is required'
        }), 400
    
    # Get content-based recommendations
    recommendations = get_similar_movies(df, title, matrices, n=15)
    
    # Apply filters
    genre = request.args.get('genre', '')
    year_min = request.args.get('year_min', 0, type=int)
    year_max = request.args.get('year_max', 3000, type=int)
    rating_min = request.args.get('rating_min', 0, type=float)
    
    filters = {}
    if genre:
        filters['genre'] = genre
    if year_min > 0:
        filters['year_min'] = year_min
    if year_max < 3000:
        filters['year_max'] = year_max
    if rating_min > 0:
        filters['rating_min'] = rating_min
    
    if filters:
        recommendations = filter_recommendations(recommendations, filters)
    
    return jsonify({
        'status': 'success',
        'count': len(recommendations),
        'data': recommendations
    })

@app.route('/recommend/popular')
def popular_recommendations():
    n = request.args.get('n', 10, type=int)
    min_votes = request.args.get('min_votes', 1000, type=int)
    
    # Get filter parameters
    genre = request.args.get('genre', '')
    year_min = request.args.get('year_min', 0, type=int)
    year_max = request.args.get('year_max', 3000, type=int)
    
    year_range = None
    if year_min > 0 and year_max < 3000:
        year_range = (year_min, year_max)
    
    # Get popularity-based recommendations
    recommendations = get_popular_movies(df, n=n, min_votes=min_votes, year_range=year_range, genre=genre)
    
    return jsonify({
        'status': 'success',
        'count': len(recommendations),
        'data': recommendations
    })

@app.route('/recommend/collaborative')
def collaborative_recommendations():
    user_id = request.args.get('user_id', 'guest')
    method = request.args.get('method', 'svd')  # 'knn', 'svd', or 'pearson'
    n = request.args.get('n', 10, type=int)
    
    # Convert user_id to int if it's a digit
    if user_id.isdigit():
        user_id = int(user_id)
        if user_id >= len(user_item_matrix):
            user_id = 0  # Default to first user if invalid
    else:
        user_id = 0  # Default to first user for guests
    
    # Get collaborative recommendations based on method
    if method == 'knn':
        recommendations = get_knn_recommendations(df, user_id, user_item_matrix, n=n)
    elif method == 'pearson':
        recommendations = get_pearson_recommendations(df, user_id, user_item_matrix, n=n)
    else:  # default to SVD
        recommendations = get_svd_recommendations(df, user_id, user_item_matrix, n=n)
    
    # Apply filters
    genre = request.args.get('genre', '')
    year_min = request.args.get('year_min', 0, type=int)
    year_max = request.args.get('year_max', 3000, type=int)
    rating_min = request.args.get('rating_min', 0, type=float)
    
    filters = {}
    if genre:
        filters['genre'] = genre
    if year_min > 0:
        filters['year_min'] = year_min
    if year_max < 3000:
        filters['year_max'] = year_max
    if rating_min > 0:
        filters['rating_min'] = rating_min
    
    if filters:
        recommendations = filter_recommendations(recommendations, filters)
    
    return jsonify({
        'status': 'success',
        'count': len(recommendations),
        'data': recommendations
    })

@app.route('/recommend/personalized')
def personalized_recommendations():
    user_id = request.args.get('user_id', 'guest')
    n = request.args.get('n', 10, type=int)
    
    # Use user interaction history if available, otherwise use default
    user_history = []
    if user_id in user_interactions:
        user_history = user_interactions[user_id]['viewed'] + user_interactions[user_id]['liked']
        # Add titles from rated movies
        user_history.extend(user_interactions[user_id]['rated'].keys())
    
    # If no history, return demographic recommendations
    if not user_history:
        # Get demographic recommendations
        genre = request.args.get('genre', '')
        year_min = request.args.get('year_min', 0, type=int)
        year_max = request.args.get('year_max', 3000, type=int)
        rating_min = request.args.get('rating_min', 0, type=float)
        certificate = request.args.get('certificate', '')
        
        filters = {}
        if genre:
            filters['genre'] = genre
        if year_min > 0:
            filters['year_min'] = year_min
        if year_max < 3000:
            filters['year_max'] = year_max
        if rating_min > 0:
            filters['rating_min'] = rating_min
        if certificate:
            filters['certificate'] = certificate
        
        recommendations = get_demographic_recommendations(df, n=n, filters=filters)
    else:
        # Get personalized recommendations
        recommendations = get_personalized_recommendations(df, user_history, matrices, n=n)
    
    return jsonify({
        'status': 'success',
        'count': len(recommendations),
        'data': recommendations
    })

@app.route('/interact', methods=['POST'])
def track_user_interaction():
    data = request.json
    user_id = data.get('user_id', 'guest')
    movie_title = data.get('title', '')
    interaction_type = data.get('type', 'view')  # 'view', 'like', 'rate'
    rating = data.get('rating')
    
    if not movie_title:
        return jsonify({
            'status': 'error',
            'message': 'Movie title is required'
        }), 400
    
    # Track the interaction
    track_interaction(user_id, movie_title, interaction_type, rating)
    
    return jsonify({
        'status': 'success',
        'message': f'Interaction of type {interaction_type} recorded'
    })

@app.route('/user/history')
def get_user_history():
    user_id = request.args.get('user_id', 'guest')
    
    if user_id not in user_interactions:
        return jsonify({
            'status': 'error',
            'message': 'User has no interaction history'
        }), 404
    
    return jsonify({
        'status': 'success',
        'data': user_interactions[user_id]
    })

@app.route('/export')
def export_data():
    format_type = request.args.get('format', 'json').lower()
    title = request.args.get('title', '')
    user_id = request.args.get('user_id', 'guest')
    
    # Get recommendations
    if title:
        recommendations = get_similar_movies(df, title, matrices, n=20)
    elif user_id in user_interactions:
        user_history = user_interactions[user_id]['viewed'] + user_interactions[user_id]['liked']
        recommendations = get_personalized_recommendations(df, user_history, matrices, n=20)
    else:
        recommendations = get_popular_movies(df, n=20)
    
    # Export to requested format
    filename = f"recommendations_{user_id}"
    file_path = export_recommendations(recommendations, format=format_type, filename=filename)
    
    if file_path:
        return jsonify({
            'status': 'success',
            'message': f'Recommendations exported as {format_type}',
            'file': file_path
        })
    else:
        return jsonify({
            'status': 'error',
            'message': f'Failed to export recommendations as {format_type}'
        }), 400

# Serve static files (frontend)
@app.route('/static/<path:path>')
def serve_static(path):
    return send_from_directory(app.static_folder, path)

# Fallback route to serve index.html for client-side routing
@app.route('/', defaults={'path': ''})
@app.route('/<path:path>')
def serve_index(path):
    if path and '.' in path:
        # Try to serve as a static file
        try:
            return send_from_directory(app.static_folder, path)
        except:
            pass
    # Serve index.html for all other routes
    return send_from_directory(app.static_folder, 'index.html')

if __name__ == '__main__':
    app.run(debug=True)