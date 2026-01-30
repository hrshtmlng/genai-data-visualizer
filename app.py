from flask import Flask
import os
from dotenv import load_dotenv
from google import genai

load_dotenv()

app = Flask(__name__)

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

@app.route("/")
def home():
    return "App running"

@app.route("/test-gemini")
def test_gemini():
    response = client.models.generate_content(
        model="models/gemini-flash-latest",
        contents="Explain a bar chart in one sentence."
    )
    return response.text


if __name__ == "__main__":
    app.run(debug=True)
