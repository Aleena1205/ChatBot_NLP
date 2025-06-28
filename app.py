from flask import Flask, request, jsonify, send_from_directory
import pandas as pd
import nltk
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
import os

# Download tokenizer
nltk.download('punkt')

# Load or train model
model_path = 'chatbot_model.pkl'
if os.path.exists(model_path):
    model = joblib.load(model_path)
else:
    data = pd.read_csv('chatbot_dataset.csv')
    data['Question'] = data['Question'].apply(lambda x: ' '.join(nltk.word_tokenize(str(x).lower())))
    model = make_pipeline(TfidfVectorizer(), MultinomialNB())
    model.fit(data['Question'], data['Answer'])
    joblib.dump(model, model_path)

def get_response(question):
    try:
        processed = ' '.join(nltk.word_tokenize(question.lower()))
        return model.predict([processed])[0]
    except Exception as e:
        return "Sorry, I couldn't understand your question."

# Initialize Flask
app = Flask(__name__, static_folder='static')

@app.route('/')
def serve_index():
    return send_from_directory(app.static_folder, 'index.html')

@app.route('/api/chatbot', methods=['POST'])
def chatbot_api():
    data = request.get_json()
    question = data.get('question', '').strip()
    if not question:
        return jsonify({'answer': 'Please enter a valid question.'})
    answer = get_response(question)
    return jsonify({'answer': answer})

if __name__ == '__main__':
    app.run(debug=True)
