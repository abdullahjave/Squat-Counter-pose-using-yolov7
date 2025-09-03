# AI-Powered Squat Counter using YOLOv7 Pose Estimation

An intelligent squat counting system that uses YOLOv7 pose estimation to automatically detect and count squats in real-time from video input or webcam feed.

## 🎯 Features

- **Real-time Squat Detection**: Automatically detects and counts squats using pose estimation
- **Angle-based Counting**: Uses hip-knee-ankle angle calculation for accurate squat detection
- **Multiple Input Sources**: Supports video files, webcam, and image sequences
- **Visual Feedback**: Real-time display of squat count and current stage (Stand/Down)
- **Customizable Angle Threshold**: Adjustable squat angle detection (default: 80°)
- **GPU Acceleration**: CUDA support for faster inference
- **Output Saving**: Save processed videos with squat counting overlay

## 🏗️ Project Structure

```
squat-counter/
├── squat_counter.py          # Main squat counting application
├── detect_pose.py            # General pose detection script
├── keypoint.ipynb           # Jupyter notebook for keypoint analysis
├── yolov7-w6-pose.pt        # Pre-trained YOLOv7 pose estimation model
├── inference/               # Input files for testing
│   ├── squat1.mp4          # Sample squat video
│   └── image.jpg           # Sample image
├── models/                  # Model architecture files
│   ├── __init__.py
│   ├── common.py           # Common model components
│   ├── experimental.py     # Model loading and experimental features
│   └── yolo.py            # YOLO model implementation
├── utils/                   # Utility functions
│   ├── datasets.py         # Data loading utilities
│   ├── general.py          # General utility functions
│   ├── plots.py           # Visualization utilities
│   ├── torch_utils.py     # PyTorch utilities
│   ├── google_utils.py    # Model download utilities
│   ├── aws/               # AWS deployment utilities
│   ├── google_app_engine/ # Google Cloud deployment
│   └── wandb_logging/     # Weights & Biases logging
└── runs/                   # Output directory for results
    └── detect/            # Detection results
        ├── exp/           # Experiment folders
        ├── exp2/
        └── ...
```

## 🚀 Quick Start

### Prerequisites

- Python 3.7+
- PyTorch 1.7+
- OpenCV
- NumPy
- CUDA (optional, for GPU acceleration)

### Installation

1. Clone the repository:
```bash
git clone https://github.com/abdullahjave/Squat-Counter-pose-using-yolov7.git
cd Squat-Counter-pose-using-yolov7
```

2. Install required dependencies:
```bash
pip install torch torchvision opencv-python numpy
```

3. Download the pre-trained model (if not included):
   - The `yolov7-w6-pose.pt` model should be in the root directory

### Usage

#### Basic Usage - Video File
```bash
python squat_counter.py --weights yolov7-w6-pose.pt --source inference/squat1.mp4 --kpt-label --view-img
```

#### Webcam Input
```bash
python squat_counter.py --weights yolov7-w6-pose.pt --source 0 --kpt-label --view-img
```

#### Custom Angle Threshold
```bash
python squat_counter.py --weights yolov7-w6-pose.pt --source inference/squat1.mp4 --kpt-label --view-img --angle 70
```

## 📋 Command Line Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--weights` | `yolov7-w6-pose.pt` | Path to model weights |
| `--source` | `0` | Input source (video file, webcam, or directory) |
| `--img-size` | `640` | Inference image size |
| `--conf-thres` | `0.25` | Object confidence threshold |
| `--iou-thres` | `0.45` | IOU threshold for NMS |
| `--device` | `''` | CUDA device (e.g., '0' or 'cpu') |
| `--view-img` | `False` | Display results in real-time |
| `--nosave` | `False` | Don't save images/videos |
| `--project` | `runs/detect` | Save results to project/name |
| `--name` | `exp` | Save results to project/name |
| `--line-thickness` | `3` | Bounding box thickness |
| `--hide-labels` | `False` | Hide labels |
| `--kpt-label` | `False` | Use keypoint labels |
| `--angle` | `80` | Squat detection angle threshold |

## 🔧 How It Works

### 1. Pose Detection
- Uses YOLOv7-W6-Pose model to detect human keypoints
- Identifies 17 key body points including hip, knee, and ankle joints

### 2. Squat Detection Algorithm
- **Keypoint Extraction**: Extracts hip (11/12), knee (13/14), and ankle (15/16) coordinates
- **Angle Calculation**: Computes the angle between hip-knee-ankle using trigonometry
- **State Machine**: Tracks squat stages:
  - **Stand**: Angle > 160° (standing position)
  - **Down**: Angle < threshold (default 80°, squatting position)
- **Counting Logic**: Increments counter when transitioning from "Down" to "Stand"

### 3. Visual Output
- Real-time overlay showing:
  - Squat counter
  - Current stage (Stand/Down) with color coding
  - Pose keypoints and skeleton
  - Bounding boxes around detected persons

## 🎨 Key Components

### Core Functions

- **`calculate_angle(coords)`**: Calculates the angle between three points (hip-knee-ankle)
- **`get_coords(kpts)`**: Extracts relevant keypoints for squat detection
- **`count_squat(angle, stage, counter)`**: Main counting logic with state management
- **`detect(opt)`**: Main detection and processing loop

### Model Architecture
- **YOLOv7-W6-Pose**: Large-scale pose estimation model
- **Input Size**: 640x640 pixels (configurable)
- **Output**: Bounding boxes + 17 keypoints per person
- **Inference Speed**: Real-time capable with GPU acceleration

## 📊 Performance

- **Accuracy**: High precision squat detection using angle-based method
- **Speed**: Real-time processing on modern GPUs
- **Robustness**: Works with various camera angles and lighting conditions
- **Multi-person**: Can detect multiple people simultaneously

## 🛠️ Customization

### Adjusting Squat Detection
- Modify the `--angle` parameter to change sensitivity
- Lower values (60-70°) for deeper squats
- Higher values (80-90°) for partial squats

### Adding New Exercises
The framework can be extended for other exercises by:
1. Identifying relevant keypoints
2. Defining angle calculations
3. Implementing exercise-specific counting logic

## 🚀 Deployment Options

### Local Development
- Run directly with Python for development and testing

### Cloud Deployment
- **AWS**: Utilities provided in `utils/aws/`
- **Google Cloud**: Configuration in `utils/google_app_engine/`
- **Docker**: Dockerfile available for containerization

### Edge Deployment
- Optimized for edge devices with model quantization
- ONNX export support for cross-platform deployment

## 📝 Output

### Console Output
```
Counter : 5
Stage: Stand
Done. (0.045s)
```

### Video Output
- Processed video saved in `runs/detect/exp/`
- Includes squat counter overlay and pose visualization
- Original video resolution maintained

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is based on YOLOv7 and follows the same licensing terms.

## 🙏 Acknowledgments

- **YOLOv7**: Original pose estimation model
- **OpenCV**: Computer vision library
- **PyTorch**: Deep learning framework


## 📞 Support

For issues and questions:
1. Check the [existing issues](https://github.com/abdullahjave/Squat-Counter-pose-using-yolov7/issues)
2. Create a new issue with detailed description
3. Include system information and error logs

## 🔗 Repository

**GitHub**: [https://github.com/abdullahjave/Squat-Counter-pose-using-yolov7](https://github.com/abdullahjave/Squat-Counter-pose-using-yolov7)

---

**Note**: This project is designed for fitness tracking and educational purposes. For professional fitness assessment, consult with qualified trainers.