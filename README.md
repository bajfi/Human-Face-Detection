# Face Detection Application

A modern, high-performance face detection application built with C++23, OpenCV, and Qt5. This application provides real-time face detection capabilities using both traditional Haar Cascade classifiers and modern DNN-based models like YuNet.

<p align="left">
    <img src="https://img.shields.io/badge/License-MIT-blue.svg" alt="License"/>
    <img src="https://img.shields.io/badge/C%2B%2B-23-blue.svg" alt="C++23"/>
    <img src="https://img.shields.io/badge/OpenCV-4.x-green.svg" alt="OpenCV"/>
    <img src="https://img.shields.io/badge/Qt-5.x-brightgreen.svg" alt="Qt"/>
</p>

## ✨ Features

### Core Functionality

- **Multi-Algorithm Support**: Haar Cascade and DNN-based face detection
- **Real-time Processing**: Live camera feed face detection
- **Video File Support**: Process and analyze video files (MP4, AVI, MOV, MKV, WMV, FLV, WebM, M4V)
- **Image Processing**: Static image face detection with multiple format support
- **Performance Monitoring**: Real-time FPS tracking and detailed performance analytics

### Advanced Features

- **Smart Model Selection**: Automatic detector selection based on file format
- **CUDA Acceleration**: GPU acceleration support when OpenCV is compiled with CUDA
- **Adaptive Memory Management**: Intelligent memory pooling for optimal performance
- **Comprehensive Logging**: Multi-level logging system with configurable output
- **Configuration Management**: Persistent settings and user preferences
- **Test Suite**: Comprehensive unit and integration tests

### User Interface

- **Modern Qt5 GUI**: Intuitive and responsive user interface
- **Performance Dashboard**: Real-time charts and metrics visualization
- **Video Playback Controls**: Play, pause, and seek through video files
- **Model Information Display**: Detailed model status and configuration

### Key Components

- **IFaceDetector**: Abstract interface for face detection implementations
- **CascadeFaceDetector**: Haar Cascade classifier implementation
- **YunetFaceDetector**: Modern DNN-based face detection
- **FaceDetectorFactory**: Smart factory for automatic model selection
- **VideoFaceDetector**: Video processing and analysis engine
- **PerformanceMonitor**: Real-time performance tracking and analytics
- **AdaptiveMemoryPool**: Intelligent memory management
- **LoggerManager**: Comprehensive logging system

## 🚀 Getting Started

### Prerequisites

- **Compiler**: GCC 14+ or Clang 17+ (C++23 support required)
- **CMake**: Version 3.12 or higher
- **Qt5**: Core, Widgets, Gui, Charts components
- **OpenCV**: 4.5+ with core, imgproc, objdetect, imgcodecs, videoio, highgui, dnn modules
- **spdlog**: Modern logging library

#### Ubuntu/Debian Installation

```bash
sudo apt update
sudo apt install build-essential cmake
sudo apt install qtbase5-dev qtcharts5-dev
sudo apt install libopencv-dev
sudo apt install libspdlog-dev
```

#### Optional: CUDA Support

For GPU acceleration, ensure OpenCV is compiled with CUDA support:

```bash
# Check CUDA availability
pkg-config --modversion opencv4
# Look for CUDA modules in OpenCV build info
```

### Building

1. **Clone the repository**:

```bash
git clone <repository-url>
cd "Face Detection"
```

2. **Create build directory**:

```bash
mkdir build && cd build
```

3. **Configure with CMake**:

```bash
cmake .. -DCMAKE_BUILD_TYPE=Release
```

4. **Build the application**:

```bash
make -j$(nproc)
```

5. **Run tests** (optional):

```bash
make test
```

### Running

Execute the application from the build directory:

```bash
./HFaceDetector
```

#### Command Line Options

```bash

# Enable debug logging
./HFaceDetector --debug

# Set specific log level
./HFaceDetector --log-level info

# Quiet mode (errors only)
./HFaceDetector --quiet

# Show help
./HFaceDetector --help
```

## 📖 Usage Guide

### Loading Models

1. **Automatic Detection**: The application automatically detects model type based on file extension
   - `.xml`, `.cascade` → Haar Cascade Detector
   - `.onnx`, `.pb`, `.tflite` → DNN Detector

2. **Supported Models**:
   - **Haar Cascades**: `haarcascade_frontalface_alt.xml` (included)
   - **YuNet ONNX**: `face_detection_yunet_2023mar.onnx` (included)

### Processing Media

#### Camera Feed

1. Click "Start Camera" to begin real-time detection
2. Select detection model from the dropdown
3. Monitor performance metrics in real-time
4. Click "Stop Camera" to end session

#### Video Files

1. Click "Load Image/Video"
2. Select video file (MP4, AVI, MOV, etc.)
3. Use "Play Video"/"Stop Video" controls
4. Videos automatically loop for continuous analysis

#### Static Images

1. Click "Load Image/Video"
2. Select image file (JPG, PNG, BMP, etc.)
3. Face detection runs automatically
4. Results displayed with bounding boxes

### Performance Monitoring

Access detailed performance analytics via the "Performance" button:

- Real-time FPS monitoring
- Average processing times
- Memory usage statistics
- Detection accuracy metrics
- Hardware utilization (GPU/CPU)

## 🧪 Testing

The project includes comprehensive test suites:

```bash
# Run all tests
cd build && make test

# Run specific test categories
./tests/test_cascade_detector
./tests/test_dnn_detector
./tests/test_integration
```

### Test Coverage

- **Unit Tests**: Individual component testing
- **Integration Tests**: Cross-component functionality
- **Performance Tests**: Benchmark and regression testing
- **Model Validation**: Accuracy and compatibility testing

## 🔧 Configuration

### Model Configuration

Models are automatically loaded from the `models/` directory:

- `models/haarcascade_frontalface_alt.xml`
- `models/face_detection_yunet_2023mar.onnx`

### Performance Tuning

Adjust detection parameters for optimal performance:

```cpp
// Example: Configure detection sensitivity
detector->setDetectionParams(
    1.1,                    // Scale factor
    3,                      // Min neighbors
    cv::Size(30, 30),      // Min size
    cv::Size(300, 300)     // Max size
);
```

### Logging Configuration

Configure logging levels via command line or code:

- `debug`: Detailed debugging information
- `info`: General information (default)
- `warning`: Warning messages
- `error`: Error messages only
- `critical`: Critical errors only

### Development Setup

```bash
# Install development dependencies
sudo apt install clang-format clang-tidy

# Format code before committing
clang-format -i src/*.cpp src/*.h

# Run static analysis
clang-tidy src/*.cpp -- -I/usr/include/opencv4
```

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **OpenCV Team**: For the comprehensive computer vision library
- **Qt Project**: For the excellent GUI framework
- **spdlog**: For the high-performance logging library
- **YuNet Authors**: For the modern face detection model
