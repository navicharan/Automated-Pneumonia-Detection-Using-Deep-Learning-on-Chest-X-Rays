from flask import Flask, request, jsonify, render_template, send_file
import os
import numpy as np
import tensorflow as tf
import cv2
import matplotlib.pyplot as plt

app = Flask(__name__, template_folder="templates")

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# Load trained model
MODEL_PATH = "best_model.h5"
model = tf.keras.models.load_model(MODEL_PATH)
model.build(input_shape=(None, 64, 64, 1))
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Function to preprocess image
def preprocess_image(image_path):
    img = cv2.imread(image_path)  # Load as RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
    img = cv2.resize(img, (64, 64))  # Resize to model input size
    img = np.expand_dims(img, axis=-1)  # Add channel dimension (1 channel for grayscale)
    img = img / 255.0  # Normalize
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    return img


# **Grad-CAM Function**
def generate_gradcam(image_path):
    img_array = preprocess_image(image_path)
    # Access the last convolutional layer
    model.summary()
    # Get the last convolutional layer in the model
    last_conv_layer = model.get_layer("conv2d_2")   #**Ensure this matches your last conv layer name**
    print(f"Last convolutional layer: {last_conv_layer.name}")
    grad_model = tf.keras.models.Model(
        [model.input], [last_conv_layer.output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        loss = predictions[:, 0]  # Target class (Pneumonia)

    # Compute gradients
    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    heatmap = tf.reduce_mean(tf.multiply(pooled_grads, conv_outputs), axis=-1)

    # Process heatmap
    heatmap = np.maximum(heatmap[0], 0)  # Apply ReLU
    heatmap /= np.max(heatmap)  # Normalize

    # Load original image
    img = cv2.imread(image_path)
    img = cv2.resize(img, (64, 64))

    # Apply heatmap
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    heatmap = np.uint8(255 * heatmap)  # Convert to 0-255
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)  # Apply colormap

    # Overlay heatmap on original image
    superimposed_img = cv2.addWeighted(img, 0.6, heatmap, 0.4, 0)

    # Save heatmap
    heatmap_path = os.path.join("uploads", "heatmap.jpg")
    cv2.imwrite(heatmap_path, superimposed_img)

    return heatmap_path

# Prediction function
def predict_pneumonia(image_path):
    img = preprocess_image(image_path)  # Preprocess the input image
    prediction = model.predict(img)  # Get model prediction
    probability = prediction[0][0]  # Extract the probability for 'pneumonia'
    if probability < 0.30:
        return "Likely Normal", probability
    elif probability < 0.70:
        return "Moderate Risk", probability
    else:
        return "High Risk", probability

# **Route for Image Upload & Prediction**
@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No selected file"}), 400

    filepath = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
    file.save(filepath)

    diagnosis, probability = predict_pneumonia(filepath)
    heatmap_path = generate_gradcam(filepath)  # Generate Grad-CAM

    return jsonify({
        "diagnosis": diagnosis,
        "pneumonia_probability": round(float(probability), 4),
        "heatmap_image": heatmap_path
    })

# **Route to Serve Heatmap Image**
@app.route("/heatmap")
def get_heatmap():
    heatmap_path = os.path.join(app.config["UPLOAD_FOLDER"], "heatmap.jpg")
    if os.path.exists(heatmap_path):
        return send_file(heatmap_path, mimetype="image/jpeg")
    return jsonify({"error": "Heatmap not found"}), 404
@app.route("/")
def index():
    return render_template("index.html")
if __name__ == "__main__":
    app.run(debug=True, port=8080)
# Print model summary to inspect input/output layers

