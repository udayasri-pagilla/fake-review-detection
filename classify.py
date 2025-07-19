
import sys
import joblib

# Load model
model = joblib.load('model/fake_genuine_model.joblib')

# Take input from command-line
if len(sys.argv) < 2:
    print("Usage: python classify.py \"your review text here\"")
    sys.exit(1)

text = sys.argv[1]

# Predict
prediction = model.predict([text])[0]
print(f"Prediction: {prediction}")