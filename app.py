from flask import Flask, request, jsonify, render_template
import os
import pickle
from flask_cors import CORS
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__, template_folder='templates')  # Explicitly setting the template folder
CORS(app)

# Load the trained TF-IDF vectorizer
with open('tfidf_vectorizer.pkl', 'rb') as f:
    vectorizer = pickle.load(f)

@app.route('/')
def home():
    return render_template('index.html')  # Serve the index.html from the templates folder

@app.route('/similarity', methods=['POST'])
def similarity():
    data = request.get_json()
    if not data or 'text1' not in data or 'text2' not in data:
        return jsonify({'error': 'Missing data, please provide text1 and text2'}), 400

    text1 = data['text1']
    text2 = data['text2']
    texts_transformed = vectorizer.transform([text1, text2])
    similarity_score = cosine_similarity(texts_transformed[0:1], texts_transformed[1:])[0][0]

    return jsonify({"similarity score": similarity_score})

if __name__ == '__main__':
    app.run(debug=True)  # Remember to turn off debug in a production environment
