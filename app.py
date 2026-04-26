from flask import Flask, render_template, request
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

# Load dataset
data = pd.read_csv("movies.csv")

# Combine features
data['combined'] = data['genre'] + " " + data['description']

# Vectorize
vectorizer = TfidfVectorizer(stop_words='english')
vectors = vectorizer.fit_transform(data['combined'])

# Similarity
similarity = cosine_similarity(vectors)

def recommend(movie_name):
    if movie_name not in data['title'].values:
        return ["Movie not found"]

    index = data[data['title'] == movie_name].index[0]
    distances = list(enumerate(similarity[index]))
    movies = sorted(distances, key=lambda x: x[1], reverse=True)[1:6]

    result = []
    for i in movies:
        result.append(data.iloc[i[0]].title)
    
    return result

@app.route('/', methods=['GET', 'POST'])
def home():
    recommendations = []
    
    if request.method == 'POST':
        movie = request.form['movie']
        recommendations = recommend(movie)
    
    return render_template('index.html', recommendations=recommendations)

if __name__ == "__main__":
    app.run(debug=True)