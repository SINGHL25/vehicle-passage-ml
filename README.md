# Vehicle Passage Prediction using ML

This is a Flask-based web app that predicts the vehicle count for the next hour using a Random Forest model trained on uploaded traffic logs.

## 🔧 Features
- Upload traffic log CSV (`timestamp`, `vehicle_count`)
- Train a RandomForest ML model
- Predict vehicle count for the next hour
- View actual vs predicted chart (Matplotlib)
- Save predictions to MongoDB Atlas (optional)

## 🗂️ Folder Structure
```
vehicle-passage-ml/
├── app.py
├── utils.py
├── model.pkl  ← Generated after first upload
├── static/
│   └── chart.png
├── templates/
│   └── index.html
├── uploads/
├── README.md
```

## 🚀 How to Run
1. Install dependencies:
   ```
   pip install flask pandas scikit-learn matplotlib pymongo
   ```

2. Update MongoDB URI in `app.py`:
   ```python
   client = MongoClient("your_mongodb_atlas_uri")
   ```

3. Run the app:
   ```
   python app.py
   ```

4. Visit `http://localhost:5000` and upload a CSV file to see the prediction and graph.

## 📊 Sample CSV Format
```
timestamp,vehicle_count
2025-06-30 08:00:00,123
2025-06-30 09:00:00,135
...
```

## ✅ To Do
- Add user login (Firebase/Auth0)
- Dockerize the app
- Deploy on Render or Fly.io
