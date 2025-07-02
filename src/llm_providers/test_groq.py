import os
import requests

def test_groq():
    """Basic test for Groq API"""
    
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        print("GROQ_API_KEY not found in environment")
        return False
    
    model = "llama3-8b-8192"
    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {"Authorization": f"Bearer {api_key}"}
    
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": "Hello, how are you?"}],
        "max_tokens": 50
    }
    
    print(f"Testing Groq API with model: {model}")
    
    try:
        response = requests.post(url, headers=headers, json=payload, timeout=30)
        print(f"Status Code: {response.status_code}")
        
        if response.status_code == 200:
            print("Success! Response received")
            try:
                result = response.json()
                if "choices" in result and len(result["choices"]) > 0:
                    message = result["choices"][0].get("message", {})
                    text = message.get("content", "")
                    print(f"Generated Text: {text}")
                else:
                    print(f"Raw Response: {result}")
                return True
            except Exception as e:
                print(f"Could not parse JSON response: {e}")
                print(f"Raw Response: {response.text}")
                return True
        else:
            print(f"Error: {response.status_code}")
            print(f"Response: {response.text}")
            return False
            
    except Exception as e:
        print(f"Request failed: {e}")
        return False

if __name__ == "__main__":
    print("Starting Groq API Test")
    print("-" * 40)
    success = test_groq()
    print("-" * 40)
    if success:
        print("Groq test completed successfully!")
    else:
        print("Groq test failed!") 