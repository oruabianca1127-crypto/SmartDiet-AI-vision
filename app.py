import os
import pandas as pd
from flask import Flask, jsonify, request
from flask_cors import CORS
from google import genai  # Official Google GenAI SDK
from google.genai import types
from rapidfuzz import process

app = Flask(__name__)
CORS(app)  # Allows your website to talk to this backend

client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))

# Load your CSV dataset once when the server starts up (Read-only reference)
df = pd.read_csv("PFD_2000_yehey.csv", encoding="latin1")
df["Food_lower"] = df["Food Names"].str.lower()

CONFIDENT_SCORE_THRESHOLD = 95
SUGGESTION_SCORE_THRESHOLD = 85


@app.route("/search", methods=["GET"])
def search_dataset():
  query = request.args.get("query", "").strip().lower()
  if not query or len(query) < 2:
    return jsonify({"success": True, "foods": []})

  # RapidFuzz match against your dataset (Read-only search, dataset cannot be edited)
  matches = process.extract(query, df["Food_lower"], limit=8)
  results = []

  for match_text, score, index in matches:
    if score >= SUGGESTION_SCORE_THRESHOLD:
      row = df.iloc[index]
      results.append({
          "description": row["Food Names"],
          "calories": float(row["Calories per gram"]) * 100,  # Per 100g serving base
          "protein": float(row["Protein per gram"]) * 100,
          "carbs": float(row["Carbohydrates per gram"]) * 100,
          "fats": float(row["Fats per gram"]) * 100,
      })

  return jsonify({"success": True, "foods": results})


@app.route("/analyze-image", methods=["POST"])
def analyze_image():
  if "image" not in request.files:
    return jsonify({"success": False, "error": "No image uploaded."}), 400

  image_file = request.files["image"]
  temp_path = "temp_food_image.jpg"
  image_file.save(temp_path)

  try:
    with open(temp_path, "rb") as f:
      image_bytes = f.read()

    prompt = (
        "Analyze this food image. Identify the primary Filipino food item"
        " present and estimate its weight in grams for a single standard"
        " serving. Return your answer *only* as foodname:grams (e.g.,"
        " adobo:150). Do not include any other text."
    )

    response = client.models.generate_content(
        model="gemini-3.8-flash",
        contents=[
            prompt,
            types.Part.from_bytes(data=image_bytes, mime_type="image/jpeg"),
        ],
        config=types.GenerateContentConfig(temperature=0.2),
    )

    ai_text = response.text.strip().lower()
  except Exception as e:
    if os.path.exists(temp_path):
      os.remove(temp_path)
    return jsonify({"success": False, "error": str(e)}), 500

  if os.path.exists(temp_path):
    os.remove(temp_path)

  parts = ai_text.split(":")
  if len(parts) != 2:
    return jsonify({"success": False, "error": "Could not parse AI response."})

  query = parts[0].strip()
  try:
    grams = float(parts[1].strip())
  except ValueError:
    grams = 100.0

  matches = process.extract(query, df["Food_lower"], limit=1)
  if not matches or matches[0][1] < 70:
    return jsonify(
        {"success": False, "error": "Food not found in dataset database."}
    )

  chosen_row = df.iloc[matches[0][2]]

  return jsonify({
      "success": True,
      "food_name": chosen_row["Food Names"],
      "calories": round(chosen_row["Calories per gram"] * grams, 2),
      "protein": round(chosen_row["Protein per gram"] * grams, 2),
      "carbs": round(chosen_row["Carbohydrates per gram"] * grams, 2),
      "fats": round(chosen_row["Fats per gram"] * grams, 2),
  })


if __name__ == "__main__":
  app.run(port=5000)