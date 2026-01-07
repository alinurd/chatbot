# app.py - SIMPLE FLASK API
from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle
import numpy as np
import json
import os

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Load trained model and components
print("🔄 Loading chatbot model...")
try:
    with open('chatbot_model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('vectorizer.pkl', 'rb') as f:
        vectorizer = pickle.load(f)
    with open('responses.pkl', 'rb') as f:
        responses = pickle.load(f)
    
    print("✅ Model loaded successfully!")
    print(f"📊 Model type: {type(model)}")
    print(f"📊 Vectorizer type: {type(vectorizer)}")
    print(f"📊 Responses count: {len(responses)}")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    model = None
    vectorizer = None
    responses = {}

# Load dataset for reference
with open('dataset.json', 'r', encoding='utf-8') as f:
    dataset = json.load(f)

# Preprocessing function (same as in training)
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import string

nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)

stop_words = set(stopwords.words('indonesian'))

def preprocess(text):
    """Preprocess user input"""
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    tokens = word_tokenize(text)
    tokens = [word for word in tokens if word not in stop_words]
    return ' '.join(tokens)

@app.route('/')
def home():
    """Home page with API info"""
    return jsonify({
        'message': 'Bengkel Motor Chatbot API',
        'status': 'running',
        'endpoints': {
            'GET /': 'API information',
            'GET /health': 'Health check',
            'POST /chat': 'Chat with bot',
            'GET /intents': 'List all intents'
        },
        'model_loaded': model is not None
    })

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy' if model is not None else 'unhealthy',
        'model_loaded': model is not None,
        'intents_count': len(responses),
        'service': 'Bengkel Motor Chatbot API'
    })

@app.route('/chat', methods=['POST'])
def chat():
    """Main chat endpoint"""
    try:
        # Get user message from request
        data = request.get_json()
        
        if not data or 'message' not in data:
            return jsonify({
                'success': False,
                'error': 'Message is required',
                'response': None
            }), 400
        
        user_message = data['message']
        
        if not user_message.strip():
            return jsonify({
                'success': False,
                'error': 'Message cannot be empty',
                'response': None
            }), 400
        
        print(f"📩 User message: {user_message}")
        
        # Preprocess user message
        processed_message = preprocess(user_message)
        print(f"🔧 Processed: {processed_message}")
        
        if not processed_message.strip():
            return jsonify({
                'success': True,
                'user_message': user_message,
                'bot_response': "Maaf, saya tidak memahami pesan Anda. Coba tanya tentang service motor, spare part, atau layanan bengkel kami.",
                'tag': 'unknown'
            })
        
        # Transform message using vectorizer
        message_vector = vectorizer.transform([processed_message])
        
        # Predict tag using model
        predicted_tag = model.predict(message_vector)[0]
        print(f"🏷️ Predicted tag: {predicted_tag}")
        
        # Get confidence score
        probabilities = model.predict_proba(message_vector)[0]
        tag_index = list(model.classes_).index(predicted_tag)
        confidence = probabilities[tag_index]
        
        print(f"📈 Confidence: {confidence:.2%}")
        
        # Get response
        if predicted_tag in responses and confidence > 0.1:  # Threshold
            response_options = responses[predicted_tag]
            bot_response = np.random.choice(response_options)
        else:
            bot_response = "Maaf, saya belum paham pertanyaan Anda. Coba tanya tentang service motor, spare part, atau layanan bengkel kami."
            predicted_tag = 'unknown'
        
        return jsonify({
            'success': True,
            'user_message': user_message,
            'bot_response': bot_response,
            'tag': predicted_tag,
            'confidence': float(confidence),
            'model_type': str(type(model).__name__)
        })
        
    except Exception as e:
        print(f"❌ Error in chat endpoint: {e}")
        return jsonify({
            'success': False,
            'error': str(e),
            'response': "Maaf, terjadi kesalahan sistem. Silakan coba lagi nanti."
        }), 500

@app.route('/intents', methods=['GET'])
def get_intents():
    """Get all available intents"""
    try:
        intents_list = []
        for intent in dataset['intents']:
            intents_list.append({
                'tag': intent['tag'],
                'patterns_count': len(intent['patterns']),
                'responses_count': len(intent['responses']),
                'sample_pattern': intent['patterns'][0] if intent['patterns'] else '',
                'sample_response': intent['responses'][0] if intent['responses'] else ''
            })
        
        return jsonify({
            'success': True,
            'total_intents': len(intents_list),
            'intents': intents_list
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 14005

@app.route('/test', methods=['GET'])
def test_chat():
    """Test endpoint with sample messages"""
    test_messages = [
        "Halo",
        "Jam berapa buka?",
        "Ganti oli berapa?",
        "Bisa booking service?",
        "Dimana lokasi bengkel?"
    ]
    
    results = []
    for msg in test_messages:
        try:
            processed = preprocess(msg)
            vector = vectorizer.transform([processed])
            tag = model.predict(vector)[0]
            confidence = model.predict_proba(vector)[0]
            tag_idx = list(model.classes_).index(tag)
            conf_score = confidence[tag_idx]
            
            results.append({
                'input': msg,
                'processed': processed,
                'predicted_tag': tag,
                'confidence': float(conf_score),
                'response': np.random.choice(responses[tag]) if tag in responses else 'No response'
            })
        except Exception as e:
            results.append({
                'input': msg,
                'error': str(e)
            })
    
    return jsonify({
        'model_test': results,
        'model_info': {
            'classes': model.classes_.tolist(),
            'class_count': len(model.classes_)
        }
    })

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 14005))
    print(f"🚀 Starting Flask API on port {port}")
    print(f"🔗 Local URL: http://localhost:{port}")
    print(f"🌐 Health check: http://localhost:{port}/health")
    print(f"💬 Chat endpoint: POST http://localhost:{port}/chat")
    print("\n📋 Available intents:")
    for intent in dataset['intents']:
        print(f"   • {intent['tag']}: {len(intent['patterns'])} patterns")
    
    app.run(host='0.0.0.0', port=port, debug=True)