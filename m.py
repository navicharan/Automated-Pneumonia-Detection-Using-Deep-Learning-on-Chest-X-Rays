from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
from PIL import Image

app = Flask(__name__)

# Load your trained model
model = tf.keras.models.load_model("best_model.h5")

@app.route("/predict", methods=["POST"])
def predict():
    file = request.files["file"]
    image = Image.open(file).resize((224, 224))  # Resize as per model input
    image = np.array(image) / 255.0  # Normalize if required
    image = np.expand_dims(image, axis=0)  # Add batch dimension

    prediction = model.predict(image)
    return jsonify({"prediction": prediction.tolist()})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
