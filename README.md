# AI Pneumonia Detection System

A web-based application that uses deep learning to detect and analyze pneumonia from chest X-ray images. The system provides detailed analysis, heatmap visualization, and nearby hospital recommendations.

## Features

- **AI-Powered Analysis**: Uses a trained CNN model to detect pneumonia from chest X-rays
- **Visual Explanations**: Generates heatmaps to highlight affected regions
- **Detailed Analysis**: Provides region-specific severity assessment and clinical interpretation
- **Hospital Locator**: Finds and displays nearby hospitals using Google Maps API
- **Secure Authentication**: Supports both Google OAuth and email/password login
- **Responsive Design**: Mobile-friendly interface built with Tailwind CSS

## Tech Stack

- **Backend**: Flask (Python)
- **Frontend**: HTML, Tailwind CSS, JavaScript
- **Database**: SQLite
- **ML Framework**: TensorFlow
- **APIs**: Google OAuth, Google Places API
- **Image Processing**: OpenCV

## Installation

1. Clone the repository:
```bash
git clone https://github.com/navicharan/Automated-Pneumonia-Detection-Using-Deep-Learning-on-Chest-X-Rays

cd pneumonia-detection
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate  # Windows
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Set up environment variables:
```bash
# Create .env file with the following:
GOOGLE_MAPS_API_KEY=your_google_maps_api_key
GOOGLE_CLIENT_ID=your_google_client_id
FLASK_SECRET_KEY=your_secret_key
```

5. Initialize the database:
```bash
flask run
# The database will be automatically created on first run
```

## Project Structure

```
├── app.py              # Main Flask application
├── model_code.py       # ML model training code
├── best_model.h5       # Trained model weights
├── requirements.txt    # Python dependencies
├── instance/          
│   └── users.db       # SQLite database
├── uploads/           # Uploaded X-ray images
├── heatmaps/          # Generated heatmap visualizations
└── src/               # Frontend templates and assets
    ├── index.html     # Main application page
    ├── login.html     # Login page
    ├── signup.html    # Registration page
    └── templates/     # Additional templates
```

## Usage

1. Start the Flask server:
```bash
python app.py
```

2. Open a browser and navigate to `http://localhost:5000`
3. Create an account or sign in with Google
4. Upload a chest X-ray image for analysis
5. View the results, heatmap, and nearby hospitals

## Features in Detail

### Image Analysis
- Probability score for pneumonia detection
- Region-specific analysis of lung zones
- Severity classification (Normal, Moderate Risk, High Risk)
- Heatmap visualization of affected areas

### Clinical Support
- Detailed anatomical analysis
- Severity assessment for different lung regions
- Clinical interpretation of findings

### Hospital Finder
- Automatic location detection
- Display of nearby hospitals
- Distance and ratings information
- Direct navigation links

## Contributing

1. Fork the repository
2. Create a new branch
3. Make your changes
4. Submit a pull request

## License

[MIT License](LICENSE)