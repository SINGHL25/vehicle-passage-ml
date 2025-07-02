from flask import Flask, render_template, request
from pymongo import MongoClient
from utils import train_model, predict_next, plot_prediction
import datetime
import os
import pickle

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# MongoDB Atlas Connection
client = MongoClient("your_mongodb_atlas_uri")  # Replace with your URI
db = client["traffic_db"]
predictions = db["predictions"]

@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    chart_path = None
    if request.method == "POST":
        file = request.files["file"]
        if file:
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(filepath)

            model, y_test, y_pred = train_model(filepath)
            plot_prediction(y_test, y_pred, "static/chart.png")

            now = datetime.datetime.now()
            hour = now.hour
            dow = now.weekday()
            pred = predict_next(model, hour, dow)

            predictions.insert_one({
                "timestamp": now.isoformat(),
                "predicted_count": int(pred)
            })

            result = f"Predicted vehicle count for hour {hour}: {int(pred)}"
            chart_path = "static/chart.png"
    return render_template("index.html", result=result, chart_path=chart_path)

if __name__ == "__main__":
    app.run(debug=True)
