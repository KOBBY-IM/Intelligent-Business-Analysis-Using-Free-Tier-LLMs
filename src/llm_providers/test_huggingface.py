import os
import requests

def test_huggingface():
    """Basic test for Hugging Face Inference API"""
    
    api_key = os.getenv("HUGGINGFACE_API_KEY")
    if not api_key:
        print("HUGGINGFACE_API_KEY not found in environment")
        return False
    
    model = "TheBloke/Llama-2-7B-Chat-GGUF"
    url = f"https://api-inference.huggingface.co/models/{model}"
    headers = {"Authorization": f"Bearer {api_key}"}
    payload = {"inputs": "Hello, how are you?"}
    
    print(f"Testing Hugging Face API with model: {model}")
    
    try:
        response = requests.post(url, headers=headers, json=payload, timeout=30)
        print(f"Status Code: {response.status_code}")
        
        if response.status_code == 200:
            print("Success! Response received")
            try:
                result = response.json()
                if isinstance(result, list) and len(result) > 0:
                    if "generated_text" in result[0]:
                        print(f"Generated Text: {result[0]['generated_text']}")
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
    print("Starting Hugging Face API Test")
    print("-" * 40)
    success = test_huggingface()
    print("-" * 40)
    if success:
        print("Hugging Face test completed successfully!")
    else:
        print("Hugging Face test failed!") 