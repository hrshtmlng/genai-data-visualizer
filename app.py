from flask import Flask, render_template, request
import os
import pandas as pd
from dotenv import load_dotenv
from google import genai

# INIT
load_dotenv()
app = Flask(__name__)

UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))


# Functions
def summarize_dataframe(df: pd.DataFrame) -> str:
    summary = []
    summary.append(f"Number of rows: {df.shape[0]}")
    summary.append(f"Number of columns: {df.shape[1]}")
    summary.append("Columns:")
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            summary.append(
                f"- {col}(numeric): min={df[col].min()}, max={df[col].max()}, mean={round(df[col].mean(), 2)}"
            )
        else:
            summary.append(
                f" - {col}(categorical): unique values={df[col].nunique()}"
            )
    return "\n".join(summary)   

def get_gemini_insights(summary_text: str) -> str:
    prompt = f"""
    You are a data analyst. Based on the following summary of a dataset, provide three key insights that could be drawn from the data, non-technical word bullet points for each insight.
    Dataset Summary:
    {summary_text}
    """
    response = client.models.generate_content(
        model = "models/gemini-flash-latest",
        contents = prompt
    )
    return response.text.strip() if response.text else "No insights generated."

def load_csv_safely(filepath):
    try:
        return pd.read_csv(filepath, encoding="utf-8", engine="python")
    except Exception:
        return pd.read_csv(
            filepath,
            encoding="latin-1",
            engine="python",
            on_bad_lines="skip"
        )

# Routes 
    
    
@app.route("/", methods=["GET","POST"])
def index():
    insights= None

    if request.method == "POST":
        file = request.files['file']
        filepath = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(filepath)

        df = load_csv_safely(filepath)
        summary_text = summarize_dataframe(df)
        insights = get_gemini_insights(summary_text)
    return render_template("index.html", insights=insights)


if __name__ == "__main__":
    app.run(debug=True)
