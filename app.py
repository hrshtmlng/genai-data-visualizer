from flask import Flask, render_template, request
import os,re
import pandas as pd
from dotenv import load_dotenv
from google import genai
import matplotlib.pyplot as plt

# INIT
load_dotenv()
app = Flask(__name__)

UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
STATIC_FOLDER = "static"
os.makedirs(STATIC_FOLDER, exist_ok=True)


# Functions
def summarize_dataframe(df: pd.DataFrame) -> str:
    summary = []

    categorical_cols = []
    numeric_cols = []

    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            numeric_cols.append(col)
        else:
            categorical_cols.append(col)

    if categorical_cols:
        for col in categorical_cols:
            value_counts = df[col].value_counts()
            summary.append(
                f"The values in one category are spread across {len(value_counts)} groups, "
                f"with most groups containing between {value_counts.min()} and {value_counts.max()} records."
            )

    if numeric_cols:
        for col in numeric_cols:
            summary.append(
                f"A numeric measure shows values ranging from {df[col].min()} to {df[col].max()}, "
                f"with an average around {round(df[col].mean(), 2)}."
            )

    return " ".join(summary)
  

def get_gemini_insights(summary_text: str) -> str:
    prompt = f"""
    You are a senior product analyst.

    Based only on the behavioral patterns implied below, write 3 concise insights.

    Rules:
    - Do NOT mention columns, identifiers, dimensions, rows, or schema
    - Do NOT describe the dataset
    - Only state implications, risks, or opportunities
    - Each insight must be one sentence
    - Use plain, non-technical English

    Patterns:
    {summary_text}
    """



    response = client.models.generate_content(
        model = "models/gemini-flash-latest",
        contents = prompt
    )


    text = response.text or ""

    # Normalize whitespace
    text = text.strip()

    # Split on numbered patterns like "1. ", "2. "
    parts = re.split(r'\s(?=\d+\.\s)', text)

    clean_lines = []

    for part in parts:
        # Remove bullets, numbers, extra symbols
        line = re.sub(r'^[\s\-\*\•]*\d*\.?\s*', '', part).strip()
        if line and line not in clean_lines:
            clean_lines.append(line)

    # Keep only first 3 insights
    clean_lines = clean_lines[:3]

    # Re-number cleanly
    final = [f"{i+1}. {line}" for i, line in enumerate(clean_lines)]

    return "\n".join(final)


plt.style.use("seaborn-v0_8-whitegrid")

def infer_graph_types(df):
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    categorical_cols = df.select_dtypes(exclude="number").columns.tolist()

    graphs = []

    # Case 1: Categorical + Numeric → Bar chart
    if categorical_cols and numeric_cols:
        graphs.append({
            "type": "bar",
            "x": categorical_cols[0],
            "y": numeric_cols[0],
            "reason": "Compare average values across categories"
        })

    # Case 2: Single numeric → Histogram
    if len(numeric_cols) == 1:
        graphs.append({
            "type": "histogram",
            "x": numeric_cols[0],
            "reason": "Understand value distribution"
        })

    # Case 3: Two numeric → Scatter
    if len(numeric_cols) >= 2:
        graphs.append({
            "type": "scatter",
            "x": numeric_cols[0],
            "y": numeric_cols[1],
            "reason": "Explore relationship between numeric variables"
        })

    return graphs



def generate_graph(df, graph, output_path):
    plt.figure()

    if graph["type"] == "bar":
        df.groupby(graph["x"])[graph["y"]].mean().plot(kind="bar")

    elif graph["type"] == "histogram":
        df[graph["x"]].plot(kind="hist")

    elif graph["type"] == "scatter":
        plt.scatter(df[graph["x"]], df[graph["y"]])
        plt.xlabel(graph["x"])
        plt.ylabel(graph["y"])

    plt.tight_layout()
    plt.savefig(output_path, dpi=120, bbox_inches="tight")
    plt.close()



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
    
    
@app.route("/", methods=["GET", "POST"])
def index():
    insights = None
    graph_paths = []

    if request.method == "POST":
        file = request.files["file"]
        filepath = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(filepath)

        # Load CSV
        df = load_csv_safely(filepath)

        # --- Insights (existing) ---
        summary_text = summarize_dataframe(df)
        insights = get_gemini_insights(summary_text)

        # --- Graph generation (new) ---
        graphs = infer_graph_types(df)

        for i, graph in enumerate(graphs[:2]):
            output_path = os.path.join("static", f"graph_{i}.png")
            generate_graph(df, graph, output_path)

            graph_paths.append({
                "path": output_path,
                "reason": graph["reason"]
            })

    return render_template(
        "index.html",
        insights=insights,
        graph_paths=graph_paths
    )


def explain_graph(graph):
    return f"This {graph['type']} chart helps to {graph['reason'].lower()}."



if __name__ == "__main__":
    app.run(debug=True)
