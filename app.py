from flask import Flask, request, jsonify, render_template, send_file, redirect, url_for, session
from google.oauth2 import id_token
from google.auth.transport import requests as google_requests  # Rename to avoid conflict
import os
import numpy as np
import tensorflow as tf
import cv2
from werkzeug.utils import secure_filename
import requests
from dotenv import load_dotenv
from functools import wraps
from werkzeug.security import check_password_hash, generate_password_hash
from datetime import datetime
from flask_sqlalchemy import SQLAlchemy

app = Flask(__name__, template_folder="src")  # Ensure Flask looks for templates

UPLOAD_FOLDER = "uploads"
HEATMAP_FOLDER = "heatmaps"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(HEATMAP_FOLDER, exist_ok=True)

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["HEATMAP_FOLDER"] = HEATMAP_FOLDER

# Add these configurations
app.secret_key = os.getenv('FLASK_SECRET_KEY', 'your-secret-key')
GOOGLE_CLIENT_ID = os.getenv('GOOGLE_CLIENT_ID')

# Load environment variables
load_dotenv()

# Get API key from environment variable
API_KEY = os.getenv('GOOGLE_MAPS_API_KEY')
if not API_KEY:
    raise ValueError("GOOGLE_MAPS_API_KEY not found in environment variables")

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
    
    # Generate heatmap and get affected regions first
    heatmap_result = generate_activation_heatmap(image_path, model)
    affected_regions = heatmap_result["affected_regions"]
    
    # Classify risk and provide relevant regions
    if probability < 0.30:
        return "Likely Normal", probability, []
    elif probability < 0.70:
        # For moderate risk, show regions with moderate or high severity
        relevant_regions = [
            region for region in affected_regions 
            if region['severity'] in ['Moderate', 'High']
        ]
        return "Moderate Risk", probability, relevant_regions
    else:
        # For high risk, show all affected regions
        return "High Risk", probability, affected_regions

def analyze_affected_regions(heatmap, threshold=0.5):
    """
    Analyzes the heatmap with anatomically specific regions and refined thresholding
    """
    # Define anatomically specific lung regions
    lung_regions = {
        'Right Upper (Apical)': ((0, 15), (0, 32)),
        'Right Middle (Cardiac)': ((15, 35), (0, 32)), 
        'Right Lower (Basal)': ((35, 64), (0, 32)),
        'Left Upper (Apical)': ((0, 15), (32, 64)),
        'Left Middle (Hilar)': ((15, 35), (32, 64)),
        'Left Lower (Basal)': ((35, 64), (32, 64))
    }
    
    # Calculate regional statistics for adaptive thresholding
    global_mean = np.mean(heatmap)
    global_std = np.std(heatmap)
    
    # Refined thresholds with anatomical considerations
    thresholds = {
        'high': global_mean + (global_std * 1.5),  # More stringent high threshold
        'moderate': global_mean + (global_std * 0.75),
        'low': global_mean + (global_std * 0.25)
    }
    
    affected_regions = []
    for region_name, ((y1, y2), (x1, x2)) in lung_regions.items():
        region_heatmap = heatmap[y1:y2, x1:x2]
        
        # Calculate regional metrics
        activation_mean = np.mean(region_heatmap)
        activation_max = np.max(region_heatmap)
        activation_area = np.sum(region_heatmap > thresholds['low']) / region_heatmap.size
        
        # Determine severity based on multiple factors
        if activation_max > thresholds['high'] and activation_area > 0.3:
            severity = "High"
            confidence = min(activation_area * 1.5, 1.0)
        elif activation_mean > thresholds['moderate'] or activation_area > 0.2:
            severity = "Moderate"
            confidence = activation_area
        elif activation_mean > thresholds['low']:
            severity = "Low"
            confidence = activation_area * 0.5
        else:
            continue  # Skip regions with minimal activation
        
        affected_regions.append({
            "name": region_name,
            "severity": severity,
            "score": float(confidence),
            "area_affected": float(activation_area),
            "max_intensity": float(activation_max)
        })
    
    # Sort by severity and then by score
    severity_order = {"High": 3, "Moderate": 2, "Low": 1}
    affected_regions.sort(key=lambda x: (severity_order[x['severity']], x['score']), reverse=True)
    
    return affected_regions

def generate_activation_heatmap(image_path, model, layer_name="conv2d_2"):
    """
    Generates and analyzes heatmap showing affected lung regions
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

    # Analyze affected regions
    affected_regions = analyze_affected_regions(heatmap)
    
    # Generate analysis text
    analysis_text = generate_analysis_text(affected_regions)
    
    return {
        "heatmap_path": heatmap_path,
        "affected_regions": affected_regions,
        "analysis": analysis_text
    }

def generate_analysis_text(affected_regions):
    """
    Generates detailed anatomical analysis of affected regions
    """
    if not affected_regions:
        return "No significant abnormalities detected in any lung region."
    
    analysis = "Detailed Regional Analysis:\n\n"
    
    # Group regions by severity
    severity_groups = {
        "High": [],
        "Moderate": [],
        "Low": []
    }
    
    for region in affected_regions:
        severity_groups[region['severity']].append(region)
    
    # Generate detailed analysis for each severity level
    for severity in ["High", "Moderate", "Low"]:
        regions = severity_groups[severity]
        if regions:
            analysis += f"{severity} Severity Regions:\n"
            for region in regions:
                analysis += f"• {region['name']}:\n"
                analysis += f"  - Affected area: {region['area_affected']*100:.1f}% of region\n"
                analysis += f"  - Intensity: {region['max_intensity']:.2f}\n"
            analysis += "\n"
    
    # Add clinical interpretation
    analysis += "Clinical Interpretation:\n"
    if severity_groups["High"]:
        right_count = sum(1 for r in severity_groups["High"] if "Right" in r["name"])
        left_count = sum(1 for r in severity_groups["High"] if "Left" in r["name"])
        
        if right_count and left_count:
            analysis += "Bilateral involvement with significant opacities. "
        elif right_count:
            analysis += "Predominant right lung involvement. "
        else:
            analysis += "Predominant left lung involvement. "
            
        if any("Lower" in r["name"] for r in severity_groups["High"]):
            analysis += "Notable basal consolidation present."
    elif severity_groups["Moderate"]:
        analysis += "Intermediate findings with patchy opacities. Consider early-stage infection or inflammation."
    else:
        analysis += "Subtle or minimal findings, possibly representing minor inflammatory changes."
    
    return analysis

# Login required decorator
def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user' not in session:
            return redirect(url_for('login_page'))  # Update this to use login_page
        return f(*args, **kwargs)
    return decorated_function

# Route to Serve the HTML Page
@app.route("/")
@login_required
def index():
    user = session.get('user')
    if not user:
        return redirect(url_for('login_page'))
    return render_template("index.html", user=user)

# Route for Image Upload & Prediction
@app.route("/predict", methods=["POST"])
@login_required
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No selected file"}), 400

    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    file.save(filepath)

    # Get Prediction with affected regions
    diagnosis, probability, affected_regions = predict_pneumonia(filepath)

    # Generate detailed analysis for affected regions
    analysis = ""
    if affected_regions:
        if diagnosis == "High Risk":
            analysis = "High Risk Areas Detected:\n"
            for region in affected_regions:
                analysis += f"• {region['name']}: {region['severity']} involvement\n"
                analysis += f"  - Affected area: {region['area_affected']*100:.1f}%\n"
                analysis += f"  - Intensity: {region['max_intensity']:.2f}\n"
        elif diagnosis == "Moderate Risk":
            analysis = "Moderate Risk Areas Detected:\n"
            for region in affected_regions:
                analysis += f"• {region['name']}: {region['severity']} involvement\n"
                analysis += f"  - Affected area: {region['area_affected']*100:.1f}%\n"

    return jsonify({
        "diagnosis": diagnosis,
        "pneumonia_probability": float(probability),
        "affected_regions": affected_regions,
        "analysis": analysis,
        "xray_url": f"/img/{filename}",
        "heatmap_url": "/heatmap"
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
        "radius": "10000",  # 10km radius
        "type": "hospital",
        "keyword": "hospital",
        "key": API_KEY  # Now using environment variable
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

# Add these new routes
@app.route('/login')
def login_page():  # Changed from 'login' to 'login_page'
    if 'user' in session:
        return redirect(url_for('index'))
    return render_template('login.html')

@app.route('/auth/google', methods=['POST'])
def google_auth():
    try:
        token = request.json['credential']
        
        # Verify the token using the correct Request object
        idinfo = id_token.verify_oauth2_token(
            token, 
            google_requests.Request(),
            GOOGLE_CLIENT_ID
        )

        # Get user info
        user_info = {
            'email': idinfo['email'],
            'name': idinfo['name'],
            'picture': idinfo['picture']
        }
        
        # Store in session
        session['user'] = user_info
        
        return jsonify({'success': True})
    except ValueError as e:
        print(f"Token validation error: {e}")
        return jsonify({'success': False, 'error': 'Invalid token'}), 401

@app.route('/auth/email', methods=['POST'])
def email_auth():
    try:
        data = request.get_json()
        email = data.get('email')
        password = data.get('password')
        remember = data.get('remember', False)

        # Find user
        user = User.query.filter_by(email=email).first()
        
        if user and check_password_hash(user.password_hash, password):
            session.permanent = remember
            session['user'] = {
                'email': user.email,
                'name': user.name,
                'picture': f'https://ui-avatars.com/api/?name={user.name.replace(" ", "+")}'
            }
            return jsonify({'success': True})
        else:
            return jsonify({
                'success': False,
                'error': 'Invalid email or password'
            }), 401

    except Exception as e:
        print(f"Login error: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/signup', methods=['POST'])
def signup():
    try:
        data = request.get_json()
        email = data.get('email')
        password = data.get('password')
        name = data.get('name')

        # Validate input
        if not all([email, password, name]):
            return jsonify({
                'success': False,
                'error': 'All fields are required'
            }), 400

        # Check if user already exists
        if User.query.filter_by(email=email).first():
            return jsonify({
                'success': False,
                'error': 'Email already registered'
            }), 400

        # Create new user
        password_hash = generate_password_hash(password)
        new_user = User(email=email, password_hash=password_hash, name=name)
        db.session.add(new_user)
        db.session.commit()

        return jsonify({'success': True})

    except Exception as e:
        print(f"Signup error: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/signup', methods=['GET'])
def signup_page():
    if 'user' in session:
        return redirect(url_for('index'))
    return render_template('signup.html')

@app.route('/logout')
def logout():
    session.pop('user', None)
    return redirect(url_for('login_page'))  # Update this to use login_page

app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///users.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(200), nullable=False)
    name = db.Column(db.String(100), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

# Initialize database
with app.app_context():
    db.create_all()

if __name__ == "__main__":
    app.run(debug=True, port=5000)
