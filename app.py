import openai
import os
from PIL import Image
import base64

# Set your OpenAI API key
openai.api_key = "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"

def encode_image(image_path):
    """Encodes the image as base64 to send to OpenAI API."""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

def analyze_image(image_path):
    """Sends the image to OpenAI's GPT-4 Vision API for analysis."""
    image_data = encode_image(image_path)
    
    response = openai.ChatCompletion.create(
        model="gpt-4-turbo",  # Use the latest vision-enabled GPT model
        messages=[
            {"role": "system", "content": "You are an AI that analyzes images and provides detailed descriptions."},
            {"role": "user", "content": [
                {"type": "text", "text": "give the name of the food you see in one word as in its name."},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_data}"}}
            ]}
        ],
        max_tokens=200
    )

    description = response["choices"][0]["message"]["content"]
    print(f"{description}")
    return description

if __name__ == "__main__":
    image_path = "apple.png"
    if os.path.exists(image_path):
        analyze_image(image_path)
    else:
        print("File not found. Please check the path and try again.")