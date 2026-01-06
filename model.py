import json
import pickle
import nltk
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import string

print("="*60)
print("🤖 TRAINING CHATBOT MODEL")
print("="*60)

# Download NLTK
try:
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    print("✅ NLTK packages ready")
except:
    print("⚠️ NLTK download skipped")

# Load and check dataset
print("\n📂 Loading dataset.json...")
try:
    with open('dataset.json', 'r', encoding='utf-8') as f:
        content = f.read()
        print(f"File size: {len(content)} characters")
        print(f"First 200 chars: {content[:200]}...")
        
        # Parse JSON
        data = json.loads(content)
        print(f"✅ JSON parsed successfully")
        print(f"Type: {type(data)}")
        
        if isinstance(data, list):
            print(f"📊 Dataset: {len(data)} items")
            print(f"First item: {data[0]}")
        elif isinstance(data, dict):
            print(f"📊 Dataset: dictionary with keys: {list(data.keys())}")
        else:
            print(f"❓ Unknown data type: {type(data)}")
            
except json.JSONDecodeError as e:
    print(f"❌ JSON Error: {e}")
    print("💡 Periksa format JSON!")
    exit(1)
except Exception as e:
    print(f"❌ Error: {e}")
    exit(1)

# Jika data adalah dictionary, konversi ke list
if isinstance(data, dict):
    print("\n⚠️  Dataset is a dictionary, converting to list...")
    if 'intents' in data:
        data = data['intents']
    else:
        # Assume it's already in the right format but wrapped
        data = list(data.values())
    
print(f"\n📊 Final dataset: {len(data)} intents")

# Preprocessing
stop_words = set(stopwords.words('indonesian'))

def preprocess(text):
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    tokens = word_tokenize(text)
    tokens = [word for word in tokens if word not in stop_words]
    return ' '.join(tokens)

# Prepare data
patterns = []
tags = []
responses = {}

print("\n🔧 Processing intents...")
for i, item in enumerate(data):
    try:
        if isinstance(item, dict):
            tag = item.get('tag', f'intent_{i}')
            patterns_list = item.get('patterns', [])
            responses_list = item.get('responses', [])
            
            responses[tag] = responses_list
            
            for pattern in patterns_list:
                patterns.append(preprocess(pattern))
                tags.append(tag)
                
            print(f"   {i+1:2d}. {tag}: {len(patterns_list)} patterns")
        else:
            print(f"   ⚠️  Item {i} is not a dict: {type(item)}")
    except Exception as e:
        print(f"   ❌ Error processing item {i}: {e}")

print(f"\n📈 Data summary:")
print(f"   Total patterns: {len(patterns)}")
print(f"   Unique tags: {len(set(tags))}")
print(f"   Tags: {', '.join(sorted(set(tags)))}")

if len(patterns) == 0:
    print("❌ Tidak ada data untuk training!")
    exit(1)

# Vectorization
print("\n🔢 Vectorizing...")
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(patterns)
y = np.array(tags)

print(f"   Vocabulary size: {len(vectorizer.vocabulary_)}")

# Train model
print("\n🤖 Training Naive Bayes model...")
model = MultinomialNB()
model.fit(X, y)

# Training accuracy
train_accuracy = model.score(X, y) * 100
print(f"✅ Training accuracy: {train_accuracy:.2f}%")

# Save model
print("\n💾 Saving model files...")
with open('chatbot_model.pkl', 'wb') as f:
    pickle.dump(model, f)

with open('vectorizer.pkl', 'wb') as f:
    pickle.dump(vectorizer, f)

with open('responses.pkl', 'wb') as f:
    pickle.dump(responses, f)

print("\n" + "="*60)
print("🎉 TRAINING COMPLETE!")
print("="*60)
print("📁 Files saved:")
print("   • chatbot_model.pkl")
print("   • vectorizer.pkl")
print("   • responses.pkl")
print(f"\n📊 Model info:")
print(f"   • Training samples: {len(patterns)}")
print(f"   • Intents: {len(responses)}")
print(f"   • Accuracy: {train_accuracy:.2f}%")
print("\n🚀 Start API with: python app.py")
print("="*60)