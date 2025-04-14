from flask import Flask, request, jsonify, render_template, send_file
import os
import numpy as np
import tensorflow as tf
import cv2
from werkzeug.utils import secure_filename
import requests

app = Flask(__name__, template_folder="src")  # Ensure Flask looks for templates

UPLOAD_FOLDER = "uploads"
HEATMAP_FOLDER = "heatmaps"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(HEATMAP_FOLDER, exist_ok=True)

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["HEATMAP_FOLDER"] = HEATMAP_FOLDER

# Load trained model from best_model.h5
# Note: Instead of a string, load the actual model.
MODEL_PATH = "best_model.h5"
model = tf.keras.models.load_model(MODEL_PATH)
# Explicitly build the model if needed (the input shape should match your model)
model.build((None, 64, 64, 1))
# Run a dummy prediction to ensure the model is initialized
dummy_input = np.zeros((1, 64, 64, 1))
model.predict(dummy_input)

# Image Preprocessing Function
def preprocess_image(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (64, 64))
    img = img / 255.0  # Normalize pixel values
    img = np.expand_dims(img, axis=(0, -1))  # Add batch & channel dimension
    return img

# Prediction Function
def predict_pneumonia(image_path):
    img = preprocess_image(image_path)
    probability = model.predict(img)[0][0]
    if probability < 0.30:
        return "Likely Normal", probability
    elif probability < 0.70:
        return "Moderate Risk", probability
    else:
        return "High Risk", probability

# New Heatmap Generation Function Using Activation Maps
def generate_activation_heatmap(image_path, model, layer_name="conv2d_2"):
    """
    Generates a heatmap by extracting the activation map from the specified convolutional layer.
    It averages the channel activations to create a 2D heatmap and overlays it on the original image.
    """
    # Preprocess image for prediction
    img = preprocess_image(image_path)

    # Create a model that outputs activations from the specified layer
    activation_model = tf.keras.models.Model(
        inputs=model.inputs,
        outputs=model.get_layer(layer_name).output
    )

    # Get the activation map for the input image
    activations = activation_model.predict(img)
    # activations shape: (1, h, w, channels)
    # Average the activations across channels to get a 2D map
    activation_map = np.mean(activations[0], axis=-1)

    # Normalize the activation map to [0,1]
    heatmap = np.maximum(activation_map, 0)
    max_val = np.max(heatmap) if np.max(heatmap) != 0 else 1
    heatmap /= max_val

    # Load original image for overlay
    img_orig = cv2.imread(image_path)
    img_orig = cv2.cvtColor(img_orig, cv2.COLOR_BGR2RGB)
    # Resize heatmap to match the original image dimensions
    heatmap = cv2.resize(heatmap, (img_orig.shape[1], img_orig.shape[0]))

    # Convert heatmap to an 8-bit format and apply a colormap
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    # Superimpose the heatmap on the original image
    superimposed_img = cv2.addWeighted(img_orig, 0.6, heatmap, 0.4, 0)

    # Save the final heatmap image
    heatmap_path = os.path.join(app.config["HEATMAP_FOLDER"], "output_heatmap.jpg")
    cv2.imwrite(heatmap_path, superimposed_img)
    return heatmap_path

# Route to Serve the HTML Page
@app.route("/")
def home():
    return render_template("index.html")  # Serves the upload form

# Route for Image Upload & Prediction
@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No selected file"}), 400

    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    file.save(filepath)

    # Get Prediction
    diagnosis, probability = predict_pneumonia(filepath)

    # Generate Activation Heatmap (using the new method)
    heatmap_path = generate_activation_heatmap(filepath, model)
    
    return jsonify({
        "diagnosis": diagnosis,
        "pneumonia_probability": round(float(probability), 4),
        "heatmap_url": "/heatmap",  # Fixed missing comma
        "xray_url": f"/img/{filename}"  # Added filename parameter
    })

# Route to Serve Heatmap
@app.route("/heatmap")
def serve_heatmap():
    heatmap_path = os.path.join(app.config["HEATMAP_FOLDER"], "output_heatmap.jpg")
    return send_file(heatmap_path, mimetype="image/jpeg")

# Fixed route to serve original image
@app.route("/img/<filename>")
def serve_img(filename):
    img_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    return send_file(img_path, mimetype="image/jpeg")

# Add this new route
@app.route("/nearby-hospitals")
def get_nearby_hospitals():
    # Get latitude and longitude from request parameters
    lat = request.args.get('lat')
    lng = request.args.get('lng')
    
    if not lat or not lng:
        return jsonify({"error": "Location parameters required"}), 400

    # Google Places API configuration
    url = "https://maps.googleapis.com/maps/api/place/nearbysearch/json"
    params = {
        "location": f"{lat},{lng}",
        "radius": "5000",  # 5km radius
        "type": "hospital",
        "keyword": "hospital",
        "key": ""  # Your API key
    }

    try:
        # Make request to Google Places API
        response = requests.get(url, params=params)
        response.raise_for_status()
        
        # Log the response for debugging
        print("Google Places API response:", response.json())
        
        return jsonify(response.json())
    except Exception as e:
        print(f"Error fetching hospitals: {str(e)}")
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True, port=8080)
