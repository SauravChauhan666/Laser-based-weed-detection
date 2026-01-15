# Laser-based Weed Detection System

A computer vision system that uses YOLOv8 object detection to identify weeds in agricultural images and provide precise laser targeting coordinates for automated weed elimination.

## Features

- **YOLOv8 Detection**: High-performance object detection for identifying weeds
- **Web Interface**: Flask-based web application for easy image upload and detection
- **Laser Targeting**: Converts pixel coordinates to laser system coordinates
- **Real-time Processing**: Fast inference on uploaded images
- **Annotated Output**: Visual feedback with bounding boxes and detection centers
- **Coordinate Export**: Download detection coordinates in JSON format for laser systems

## Project Structure

```
├── app.py                          # Flask web application
├── train.py                        # Model training script
├── detect.py                       # Weed detection class
├── check_model.py                  # Model status checker
├── requirements.txt                # Python dependencies
├── models/                         # Trained models directory
│   └── best.pt                    # Best trained model
├── runs/                           # Training runs and results
│   └── train/
│       └── weed_detection/
│           ├── weights/            # Model weights from training
│           ├── args.yaml          # Training arguments
│           └── results.csv        # Training metrics
├── static/                         # Web app static files
│   ├── script.js                  # Frontend JavaScript
│   ├── style.css                  # Frontend styling
│   └── results/                   # Detection results directory
├── templates/                      # HTML templates
│   ├── index.html                 # Main detection interface
│   └── result.html                # Results display page
└── uploads/                        # Uploaded images directory
```

## Installation

### Prerequisites
- Python 3.8+
- CUDA-capable GPU (optional, for faster training)

### Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/SauravChauhan666/Laser-based-weed-detection.git
   cd Laser-based-weed-detection
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Check model status**
   ```bash
   python check_model.py
   ```

## Usage

### Training the Model

To train the YOLOv8 model on your weed detection dataset:

```bash
python train.py
```

Optional arguments:
```bash
python train.py --validate  # Validate model after training
```

**Note**: Ensure you have a `data/data.yaml` file with proper dataset configuration.

### Running Detection on a Single Image

```bash
python detect.py --image path/to/image.jpg --conf 0.25 --output output.jpg
```

**Arguments**:
- `--image`: Path to input image (required)
- `--model`: Path to model file (default: `models/best.pt`)
- `--conf`: Confidence threshold (default: 0.25)
- `--output`: Output image path (default: `output.jpg`)

### Running the Web Application

Start the Flask web server:

```bash
python app.py
```

Then open your browser and navigate to:
```
http://localhost:5000
```

**Features**:
- Upload images (PNG, JPG, JPEG, BMP, TIFF)
- Adjust confidence threshold
- View annotated results with detection boxes
- Download detection coordinates for laser system

## API Endpoints

### GET `/`
Main web interface for image upload and detection

### POST `/upload`
Upload an image for weed detection
- **Parameters**: 
  - `file`: Image file (multipart/form-data)
  - `confidence`: Confidence threshold (optional, default: 0.25)
- **Response**: JSON with detections and annotated image

### POST `/download_coordinates`
Get detection coordinates in laser system format
- **Parameters**: JSON with detection data
- **Response**: Formatted coordinates for laser targeting system

### GET `/health`
Health check endpoint
- **Response**: Server status and model loading status

## Model Details

- **Architecture**: YOLOv8 Nano (YOLOv8n)
- **Input Size**: 640×640 pixels
- **Classes**: Weed (1 class)
- **Training Parameters**:
  - Epochs: 20
  - Batch Size: 16
  - Optimizer: Adam
  - Learning Rate: 0.01 (initial), 0.01 (final)
  - Image Size: 640×640

## Laser Coordinate System

The system converts pixel coordinates to laser targeting coordinates:
- **Pixel Space**: Original image coordinates
- **Laser Space**: Normalized to 0-1000 range (adjustable based on your laser system)

To customize laser coordinate conversion, modify the `calculate_laser_coordinates()` method in `detect.py`.

## Configuration

### Upload Limits
- Maximum file size: 16 MB
- Allowed formats: PNG, JPG, JPEG, BMP, TIFF

### Detection Parameters
- Default confidence threshold: 0.25
- Adjustable via web interface or API

## Troubleshooting

### Model Not Loading
```bash
python check_model.py
```
This will verify model status and copy weights if needed.

### CUDA Issues
If you encounter GPU issues, the system will automatically fall back to CPU:
```bash
# Force CPU mode by modifying device parameter in train.py
device='cpu'  # Instead of device=0
```

### Port Already in Use
Change the port in `app.py`:
```python
app.run(debug=True, host='0.0.0.0', port=5001)  # Use different port
```

## Performance Metrics

After training, check validation metrics in `runs/train/weed_detection/`:
- Mean Average Precision (mAP50)
- Precision and Recall
- Loss curves
- Confusion matrix

