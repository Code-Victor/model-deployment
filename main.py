from flask import Flask, request, jsonify
import joblib

app = Flask(__name__)

# Load the saved model
model = joblib.load("sentiment_model.pkl")
vectorizer = joblib.load("sentiment_vectorizer.pkl")


# create a route that manages user request and does sentiment prediction
@app.post("/predict")
def predict():
    """
    The `predict` function takes in a JSON object containing a text and uses a pre-trained model and
    vectorizer to predict the sentiment of the text, returning the prediction as a JSON response.
    :return: a JSON response containing the predicted sentiment. The sentiment is returned as a
    key-value pair with the key "sentiment" and the predicted sentiment value as the corresponding
    value.
    """
    data = request.get_json()
    text = data["text"]
    vectorized_text = vectorizer.transform([text])
    prediction = model.predict(vectorized_text)[0]
    return jsonify({"sentiment": prediction})


if __name__ == "__main__":
    app.run(debug=True)
