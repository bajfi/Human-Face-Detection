// Copyright (c) 2025 JackLee
//
// This software is released under the MIT License.
// https://opensource.org/licenses/MIT

#ifndef FACEDETECTIONWINDOW_H
#define FACEDETECTIONWINDOW_H

#include <QMainWindow>
#include <QLabel>
#include <QPushButton>
#include <QVBoxLayout>
#include <QHBoxLayout>
#include <QFileDialog>
#include <QMessageBox>
#include <QTimer>
#include <QApplication>
#include <QPixmap>
#include <QImage>
#include <QGroupBox>
#include <QProgressBar>
#include <QStatusBar>
#include <QCheckBox>
#include <QTextEdit>
#include <QDialog>
#include <QInputDialog>
#include <QLineEdit>
#include <QComboBox>
#include <QDateTime>

#include <opencv2/opencv.hpp>
#include <opencv2/objdetect.hpp>
#include <opencv2/imgproc.hpp>
#include "IFaceDetector.h"
#include "FaceDetectorFactory.h"
#include "VideoFaceDetector.h"
#include "PerformanceMonitor.h"
#include "ConfigurationManager.h"

/**
 * @class FaceDetectionWindow
 * @brief Main application window providing GUI for face detection operations
 *
 * This class implements the primary user interface for the face detection
 * application, providing functionality for:
 * - Loading and processing images/videos
 * - Real-time camera face detection
 * - Model selection and configuration
 * - Performance monitoring and optimization
 * - GPU acceleration control
 *
 * The window supports multiple face detection algorithms including:
 * - Haar Cascade classifiers
 * - DNN-based detectors (YuNet)
 * - CUDA-accelerated processing (when available)
 *
 * @note This class follows the Qt Model-View pattern and uses dependency
 * injection for face detection components to ensure modularity and testability.
 *
 * @see IFaceDetector, VideoFaceDetector, PerformanceMonitor
 */
class FaceDetectionWindow : public QMainWindow
{
    Q_OBJECT

  public:
    /**
     * @brief Constructs a new Face Detection Window
     * @param parent Parent widget (optional, defaults to nullptr)
     *
     * Initializes the GUI components, sets up layouts, connects signals/slots,
     * and attempts to load a default face detection model if available.
     */
    explicit FaceDetectionWindow(QWidget* parent = nullptr);

    /**
     * @brief Destructor
     *
     * Properly cleans up resources including:
     * - Stopping any active camera/video processing
     * - Releasing OpenCV resources
     * - Destroying Qt widgets and timers
     */
    ~FaceDetectionWindow();

  private slots:
    /**
     * @brief Opens file dialog to load an image or video file
     *
     * Supports common image formats (JPG, PNG, BMP) and video formats (MP4,
     * AVI). Automatically determines file type and processes accordingly.
     */
    void loadFile();

    /**
     * @brief Loads and processes a specific video file
     * @param fileName Path to the video file to load
     *
     * Initializes video capture and prepares for frame-by-frame processing.
     * Validates file format and accessibility before loading.
     */
    void loadVideoFile(const QString& fileName);

    /**
     * @brief Opens file dialog to load a face detection model
     *
     * Supports XML (Haar Cascade) and ONNX (DNN) model formats.
     * Validates model compatibility and updates detector accordingly.
     */
    void loadModel();

    /**
     * @brief Starts real-time camera face detection
     *
     * Initializes default camera device and begins continuous frame capture.
     * Enables camera timer for periodic frame processing.
     */
    void startCamera();

    /**
     * @brief Stops camera capture and face detection
     *
     * Releases camera resources and disables frame processing timer.
     * Updates UI state to reflect inactive camera status.
     */
    void stopCamera();

    /**
     * @brief Processes a single frame from camera or video
     *
     * Core processing loop that:
     * - Captures frame from active source
     * - Performs face detection
     * - Updates display with results
     * - Tracks performance metrics
     */
    void processFrame();

    /**
     * @brief Performs face detection on the current loaded image
     *
     * Applies the selected face detection algorithm to the currently
     * loaded static image and displays results with bounding boxes.
     */
    void detectFaces();

    /**
     * @brief Handles CUDA acceleration toggle
     * @param enabled True to enable GPU acceleration, false to disable
     *
     * Reconfigures face detector to use/avoid CUDA-accelerated processing.
     * Validates GPU availability before enabling acceleration.
     */
    void onCudaToggled(bool enabled);

    /**
     * @brief Shows performance monitoring dialog
     *
     * Displays detailed performance metrics including:
     * - Frame processing times
     * - Memory usage statistics
     * - GPU utilization (if applicable)
     */
    void showPerformanceDialog();

    /**
     * @brief Shows configuration settings dialog
     *
     * Allows user to modify:
     * - Detection sensitivity parameters
     * - Processing options
     * - Display preferences
     */
    void showConfigDialog();

    /**
     * @brief Toggles video enhancement on/off for loaded video files
     *
     * Enables/disables automatic frame processing for video files.
     * Updates button states and processing status accordingly.
     */
    void toggleVideoEnhancement();

    /**
     * @brief Updates performance display in the main window
     *
     * Refreshes FPS counter and other performance indicators
     * displayed in the main interface.
     */
    void updatePerformanceDisplay();

    /**
     * @brief Handles processed frame results from video detector
     * @param result Frame processing result containing detected faces and
     * metadata
     *
     * Receives results from background video processing and updates
     * the display with detected faces and performance information.
     */
    void onFrameProcessed(const VideoFaceDetector::FrameResult& result);

  private:
    // ===== Private Methods =====

    /**
     * @brief Initializes and arranges all UI components
     *
     * Creates the main layout, control panels, buttons, and display areas.
     * Connects signals and slots for user interaction handling.
     */
    void setupUI();

    /**
     * @brief Performs face detection on a given image
     * @param image Input image for face detection
     * @return Modified image with face detection results (bounding boxes)
     *
     * Applies the currently selected face detection algorithm to the input
     * image and draws bounding rectangles around detected faces.
     */
    cv::Mat detectFacesInImage(const cv::Mat& image);

    /**
     * @brief Converts OpenCV Mat to Qt QImage for display
     * @param mat OpenCV matrix to convert
     * @return QImage suitable for Qt display widgets
     *
     * Handles color space conversion (BGR to RGB) and format transformation
     * for proper display in Qt widgets.
     */
    QImage matToQImage(const cv::Mat& mat);

    /**
     * @brief Updates the main image display with processed results
     * @param image Processed image to display
     *
     * Converts OpenCV image to Qt format and updates the display label.
     * Handles scaling and aspect ratio preservation.
     */
    void updateImageDisplay(const cv::Mat& image);

    /**
     * @brief Displays error message to user
     * @param message Error message to display
     *
     * Shows a modal error dialog with the specified message.
     * Used for file loading errors, model validation failures, etc.
     */
    void showError(const QString& message);

    /**
     * @brief Updates model information display
     *
     * Refreshes the UI elements showing current model information,
     * including model type, path, and detection capabilities.
     */
    void updateModelInfo();

    /**
     * @brief Generates file filter string for model selection dialog
     * @return Filter string for supported model file formats
     *
     * Creates appropriate file filter for QFileDialog based on
     * supported model formats (XML for Haar, ONNX for DNN).
     */
    QString getModelFilterString() const;

    /**
     * @brief Attempts to load a default face detection model
     *
     * Searches for and loads a default model from the models directory.
     * Provides fallback functionality for initial application startup.
     */
    void tryLoadDefaultModel();

    // ===== UI Components =====

    /// Main central widget containing all UI elements
    QWidget* m_centralWidget;

    /// Primary vertical layout for the main window
    QVBoxLayout* m_mainLayout;

    /// Horizontal layout for control buttons
    QHBoxLayout* m_buttonLayout;

    /// Group box containing control elements
    QGroupBox* m_controlGroup;

    /// Group box containing display elements
    QGroupBox* m_displayGroup;

    // Control Buttons
    QPushButton* m_loadFileButton;    ///< Button to load image/video files
    QPushButton* m_loadModelButton;   ///< Button to load detection models
    QPushButton* m_startCameraButton; ///< Button to start camera capture
    QPushButton* m_stopCameraButton;  ///< Button to stop camera capture
    QPushButton* m_detectButton;      ///< Button to perform face detection
    QCheckBox* m_cudaCheckBox; ///< Checkbox to enable/disable CUDA acceleration
    QPushButton* m_performanceButton; ///< Button to show performance dialog
    QPushButton* m_configButton;      ///< Button to show configuration dialog
    QPushButton* m_videoProcessingButton; ///< Button to toggle video processing

    // Display Elements
    QLabel* m_imageLabel;        ///< Primary image display area
    QLabel* m_modelInfoLabel;    ///< Display for current model information
    QLabel* m_performanceLabel;  ///< Display for performance metrics
    QLabel* m_fpsLabel;          ///< Display for frames per second
    QProgressBar* m_progressBar; ///< Progress indicator for operations

    // ===== OpenCV Components =====

    /// Camera capture device for real-time processing
    cv::VideoCapture m_camera;

    /// Video file capture for batch processing
    cv::VideoCapture m_videoCapture;

    /// Currently loaded/processed image
    cv::Mat m_currentImage;

    // ===== Face Detection Components =====

    /// Smart pointer to the active face detector implementation
    FaceDetectorPtr m_faceDetector;

    /// Video-specific face detector for optimized video processing
    std::unique_ptr<VideoFaceDetector> m_videoDetector;

    /// Performance monitoring and metrics collection
    std::unique_ptr<PerformanceMonitor> m_performanceMonitor;

    /// Configuration management for application settings
    std::unique_ptr<ConfigurationManager> m_configManager;

    // ===== Timers =====

    /// Timer for periodic camera frame capture
    QTimer* m_cameraTimer;

    /// Timer for updating performance display
    QTimer* m_performanceUpdateTimer;

    // ===== Application Status =====

    /// Flag indicating if camera capture is active
    bool m_cameraActive;

    /// Flag indicating if video processing is active
    bool m_videoProcessingActive;

    /// Flag indicating if a video file is currently loaded
    bool m_videoFileActive;

    /// Path to the currently loaded detection model
    QString m_modelPath;

    /// Path to the currently loaded video file
    QString m_currentVideoPath;

    /// Type of the currently active face detector
    FaceDetectorFactory::DetectorType m_currentDetectorType;

    // ===== Performance Tracking =====

    /// Timestamp of the last processed frame for FPS calculation
    std::chrono::high_resolution_clock::time_point m_lastFrameTime;

    /// Total number of processed frames for averaging
    size_t m_frameCount;
};

#endif // FACEDETECTIONWINDOW_H
