from src.DiamondPricePrediction.pipelines.prediction_pipeline import CustomData, PredictPipeline
from flask import Flask, request, render_template, jsonify

app = Flask(__name__)

@app.route('/')
def home_page():
    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)  # debug=True will auto-restart on changes
