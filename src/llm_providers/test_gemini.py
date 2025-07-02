import os
import requests

def test_gemini():
    """Basic test for Google Gemini API"""
    
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        print("GOOGLE_API_KEY not found in environment")
        return False
    
    model = "gemini-1.5-flash"
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}"
    
    payload = {
        "contents": [
            {
                "parts": [
                    {
                        "text": "Hello, how are you?"
                    }
                ]
            }
        ]
    }
    
    print(f"Testing Gemini API with model: {model}")
    
    try:
        response = requests.post(url, json=payload, timeout=30)
        print(f"Status Code: {response.status_code}")
        
        if response.status_code == 200:
            print("Success! Response received")
            try:
                result = response.json()
                if "candidates" in result and len(result["candidates"]) > 0:
                    candidate = result["candidates"][0]
                    if "content" in candidate and "parts" in candidate["content"]:
                        text = candidate["content"]["parts"][0].get("text", "")
                        print(f"Generated Text: {text}")
                    else:
                        print(f"Raw Response: {result}")
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
    print("Starting Gemini API Test")
    print("-" * 40)
    success = test_gemini()
    print("-" * 40)
    if success:
        print("Gemini test completed successfully!")
    else:
        print("Gemini test failed!") 