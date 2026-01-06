# test_api.py
import requests
import json

BASE_URL = "http://localhost:5000"

def test_health():
    print("🧪 Testing API Health...")
    try:
        response = requests.get(f"{BASE_URL}/health")
        print(f"✅ Status Code: {response.status_code}")
        print(f"✅ Response: {json.dumps(response.json(), indent=2)}")
    except Exception as e:
        print(f"❌ Error: {e}")

def test_chat(message):
    print(f"\n💬 Testing Chat: '{message}'")
    try:
        response = requests.post(
            f"{BASE_URL}/chat",
            json={"message": message},
            headers={"Content-Type": "application/json"}
        )
        print(f"✅ Status Code: {response.status_code}")
        data = response.json()
        if data.get('success'):
            print(f"✅ User: {data['user_message']}")
            print(f"✅ Bot: {data['bot_response']}")
            print(f"✅ Tag: {data['tag']}")
            print(f"✅ Confidence: {data['confidence']:.2%}")
        else:
            print(f"❌ Error: {data.get('error')}")
    except Exception as e:
        print(f"❌ Error: {e}")

def test_intents():
    print("\n📋 Testing Intents List...")
    try:
        response = requests.get(f"{BASE_URL}/intents")
        print(f"✅ Status Code: {response.status_code}")
        data = response.json()
        if data.get('success'):
            print(f"✅ Total Intents: {data['total_intents']}")
            for intent in data['intents'][:5]:  # Show first 5
                print(f"   • {intent['tag']}: {intent['patterns_count']} patterns")
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    print("=" * 50)
    print("🤖 BENGKEL MOTOR CHATBOT API TEST")
    print("=" * 50)
    
    # Test 1: Health check
    test_health()
    
    # Test 2: List intents
    test_intents()
    
    # Test 3: Chat samples
    test_messages = [
        "Halo",
        "Jam berapa buka?",
        "Ganti oli berapa?",
        "Bisa booking service?",
        "Dimana lokasi bengkel?",
        "Ada layanan derek?"
    ]
    
    for msg in test_messages:
        test_chat(msg)
    
    print("\n" + "=" * 50)
    print("✅ All tests completed!")
    print("=" * 50)