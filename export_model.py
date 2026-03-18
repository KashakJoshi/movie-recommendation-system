import joblib

model = joblib.load("models/recommendation_model.pkl")
joblib.dump(model, "model.pkl")

print("Model exported successfully")