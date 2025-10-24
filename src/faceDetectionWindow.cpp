// Copyright (c) 2025 JackLee
//
// This software is released under the MIT License.
// https://opensource.org/licenses/MIT

#include "faceDetectionWindow.h"
#include "ModelValidator.h"
#include "logging/Logging.h"
#include "YunetFaceDetector.h"
#include "CascadeFaceDetector.h"

FaceDetectionWindow::FaceDetectionWindow(QWidget* parent)
  : QMainWindow(parent),
    m_centralWidget(nullptr),
    m_cameraActive(false),
    m_videoProcessingActive(false),
    m_videoFileActive(false),
    m_cameraTimer(new QTimer(this)),
    m_performanceUpdateTimer(new QTimer(this)),
    m_currentDetectorType(FaceDetectorFactory::DetectorType::UNKNOWN),
    m_lastFrameTime(std::chrono::high_resolution_clock::now()),
    m_frameCount(0)
{
    // Initialize enhanced components
    m_configManager = std::make_unique<ConfigurationManager>();
    m_configManager->initialize();

    PerformanceMonitor::Config perfConfig;
    perfConfig.targetFrameTime = std::chrono::milliseconds(33); // 30 FPS target
    perfConfig.enableAdaptiveResolution = true;
    perfConfig.enableAdaptiveFrameSkip = true;
    m_performanceMonitor = std::make_unique<PerformanceMonitor>(perfConfig);

    setupUI();

    // Connect timers
    connect(
      m_cameraTimer, &QTimer::timeout, this, &FaceDetectionWindow::processFrame
    );
    connect(
      m_performanceUpdateTimer,
      &QTimer::timeout,
      this,
      &FaceDetectionWindow::updatePerformanceDisplay
    );

    // Start performance monitoring
    m_performanceMonitor->start();
    m_performanceUpdateTimer->start(
      500
    ); // Update every 500ms for more responsive updates

    // Initialize timing
    m_lastFrameTime =
      std::chrono::high_resolution_clock::now(); // Set window properties
    setWindowTitle("Intelligent Face Detection - OpenCV & Qt");
    setMinimumSize(800, 600);
    resize(1000, 700);

    // Try to auto-load a default model
    tryLoadDefaultModel();

    statusBar()->showMessage(
      "Ready for intelligent face detection - Load a model to get started"
    );
}

FaceDetectionWindow::~FaceDetectionWindow()
{
    // Stop camera timer first to prevent any callback during destruction
    if (m_cameraTimer)
    {
        m_cameraTimer->stop();
    }

    // Set camera as inactive to prevent processFrame from running
    m_cameraActive = false;

    // Safely release camera resources
    if (m_camera.isOpened())
    {
        m_camera.release();
    }
    LOG_NAMED_INFO("FaceDetectionWindow", "Application exiting.");
}

void FaceDetectionWindow::setupUI()
{
    m_centralWidget = new QWidget(this);
    setCentralWidget(m_centralWidget);

    m_mainLayout = new QVBoxLayout(m_centralWidget);

    // Control group
    m_controlGroup = new QGroupBox("Controls", this);
    m_buttonLayout = new QHBoxLayout(m_controlGroup);

    m_loadFileButton = new QPushButton("Load File", this);
    m_loadModelButton = new QPushButton("Load Model", this);
    m_startCameraButton = new QPushButton("Start Camera", this);
    m_stopCameraButton = new QPushButton("Stop Camera", this);
    m_detectButton = new QPushButton("Detect Faces", this);
    m_videoProcessingButton = new QPushButton("Enhanced Video", this);
    m_performanceButton = new QPushButton("Performance", this);
    m_configButton = new QPushButton("Config", this);

    // CUDA acceleration checkbox
    m_cudaCheckBox = new QCheckBox("Enable CUDA Acceleration", this);
    bool cudaAvailable = YunetFaceDetector::isCudaAvailable() ||
                         CascadeFaceDetector::isCudaAvailable();
    m_cudaCheckBox->setChecked(cudaAvailable);
    m_cudaCheckBox->setEnabled(cudaAvailable);
    if (!cudaAvailable)
    {
        m_cudaCheckBox->setToolTip(
          "CUDA acceleration is not available on this system"
        );
    }

    m_stopCameraButton->setEnabled(false);
    m_videoProcessingButton->setToolTip(
      "Enable enhanced video processing with GPU acceleration"
    );
    m_performanceButton->setToolTip("View real-time performance metrics");
    m_configButton->setToolTip("Configure detection profiles and settings");

    m_buttonLayout->addWidget(m_loadFileButton);
    m_buttonLayout->addWidget(m_loadModelButton);
    m_buttonLayout->addWidget(m_startCameraButton);
    m_buttonLayout->addWidget(m_stopCameraButton);
    m_buttonLayout->addWidget(m_detectButton);
    m_buttonLayout->addWidget(m_videoProcessingButton);
    m_buttonLayout->addWidget(m_performanceButton);
    m_buttonLayout->addWidget(m_configButton);
    m_buttonLayout->addWidget(m_cudaCheckBox);
    m_buttonLayout->addStretch();

    // Display group
    m_displayGroup = new QGroupBox("Image Display", this);
    QVBoxLayout* displayLayout = new QVBoxLayout(m_displayGroup);

    m_imageLabel = new QLabel(this);
    m_imageLabel->setAlignment(Qt::AlignCenter);
    m_imageLabel->setMinimumSize(640, 480);
    m_imageLabel->setStyleSheet(
      "QLabel { border: 1px solid gray; background-color: #f0f0f0; }"
    );
    m_imageLabel->setText("No image loaded");

    displayLayout->addWidget(m_imageLabel);

    // Status and progress
    m_modelInfoLabel = new QLabel("Model: None loaded", this);
    m_modelInfoLabel->setStyleSheet(
      "QLabel { font-weight: bold; color: #666; }"
    );

    m_performanceLabel = new QLabel("Performance: Ready", this);
    m_performanceLabel->setStyleSheet(
      "QLabel { font-weight: bold; color: #006600; }"
    );

    m_fpsLabel = new QLabel("FPS: 0.0", this);
    m_fpsLabel->setStyleSheet("QLabel { font-weight: bold; color: #0066CC; }");

    m_progressBar = new QProgressBar(this);
    m_progressBar->setVisible(false);

    // Add all to main layout
    m_mainLayout->addWidget(m_controlGroup);
    m_mainLayout->addWidget(m_displayGroup);

    // Status layout
    QHBoxLayout* statusLayout = new QHBoxLayout();
    statusLayout->addWidget(m_modelInfoLabel);
    statusLayout->addStretch();
    statusLayout->addWidget(m_performanceLabel);
    statusLayout->addWidget(m_fpsLabel);

    QWidget* statusWidget = new QWidget(this);
    statusWidget->setLayout(statusLayout);
    m_mainLayout->addWidget(statusWidget);
    m_mainLayout->addWidget(m_progressBar);

    // Connect signals
    connect(
      m_loadFileButton,
      &QPushButton::clicked,
      this,
      &FaceDetectionWindow::loadFile
    );
    connect(
      m_loadModelButton,
      &QPushButton::clicked,
      this,
      &FaceDetectionWindow::loadModel
    );
    connect(
      m_startCameraButton,
      &QPushButton::clicked,
      this,
      &FaceDetectionWindow::startCamera
    );
    connect(
      m_stopCameraButton,
      &QPushButton::clicked,
      this,
      &FaceDetectionWindow::stopCamera
    );
    connect(
      m_detectButton,
      &QPushButton::clicked,
      this,
      &FaceDetectionWindow::detectFaces
    );
    connect(
      m_cudaCheckBox,
      &QCheckBox::toggled,
      this,
      &FaceDetectionWindow::onCudaToggled
    );
    connect(
      m_videoProcessingButton,
      &QPushButton::clicked,
      this,
      &FaceDetectionWindow::toggleVideoEnhancement
    );
    connect(
      m_performanceButton,
      &QPushButton::clicked,
      this,
      &FaceDetectionWindow::showPerformanceDialog
    );
    connect(
      m_configButton,
      &QPushButton::clicked,
      this,
      &FaceDetectionWindow::showConfigDialog
    );
}

void FaceDetectionWindow::loadModel()
{
    QString filterString = getModelFilterString();

    QString fileName = QFileDialog::getOpenFileName(
      this, tr("Load Face Detection Model"), "", filterString
    );

    if (!fileName.isEmpty())
    {
        m_progressBar->setVisible(true);
        m_progressBar->setRange(0, 0); // Indeterminate progress

        std::string modelPath = fileName.toStdString();

        // Validate the model file first
        ValidationResult validation = ModelValidator::validateModel(modelPath);
        if (!validation.isValid)
        {
            m_progressBar->setVisible(false);
            showError(
              "Model validation failed: " +
              QString::fromStdString(validation.errorMessage)
            );
            return;
        }

        // Show warning if there is one
        if (!validation.warningMessage.empty())
        {
            QMessageBox::warning(
              this,
              "Model Warning",
              QString::fromStdString(validation.warningMessage)
            );
            LOG_NAMED_WARNING(
              "FaceDetectionWindow",
              std::format(
                "Model warning for {}: {}", modelPath, validation.warningMessage
              )
            );
        }

        // Check if model format is supported
        if (!FaceDetectorFactory::isModelSupported(modelPath))
        {
            m_progressBar->setVisible(false);
            showError("Unsupported model format: " + fileName);
            return;
        }

        // Create appropriate detector using factory
        auto detector = FaceDetectorFactory::createOptimizedDetector(modelPath);

        if (!detector)
        {
            m_progressBar->setVisible(false);
            showError(
              "Failed to create detector for: " + fileName +
              "\nPlease ensure the model file is valid and OpenCV DNN module "
              "is available."
            );
            LOG_NAMED_ERROR(
              "FaceDetectionWindow",
              std::format("Failed to create detector for: {}", modelPath)
            );
            return;
        }

        // If successful, update our detector
        m_faceDetector = std::move(detector);
        m_modelPath = fileName;
        m_currentDetectorType = FaceDetectorFactory::detectModelType(modelPath);

        updateModelInfo();

        m_progressBar->setVisible(false);
        statusBar()->showMessage(
          "Model loaded: " + QFileInfo(fileName).filePath()
        );
        LOG_NAMED_INFO(
          "FaceDetectionWindow", std::format("Loaded model: {}", modelPath)
        );
    }
}

void FaceDetectionWindow::loadFile()
{
    QString fileName = QFileDialog::getOpenFileName(
      this,
      tr("Open Image or Video"),
      "",
      tr(
        "Media Files (*.png *.jpg *.jpeg *.bmp *.tiff *.tif *.mp4 *.avi *.mov "
        "*.mkv *.wmv *.flv);;Image Files (*.png *.jpg *.jpeg *.bmp *.tiff "
        "*.tif);;Video Files (*.mp4 *.avi *.mov *.mkv *.wmv *.flv);;All Files "
        "(*)"
      )
    );

    if (!fileName.isEmpty())
    {
        QFileInfo fileInfo(fileName);
        QString extension = fileInfo.suffix().toLower();

        // Check if it's a video file
        QStringList videoExtensions = {
          "mp4", "avi", "mov", "mkv", "wmv", "flv", "webm", "m4v"
        };

        if (videoExtensions.contains(extension))
        {
            loadVideoFile(fileName);
        }
        else
        {
            // Load as image
            LOG_NAMED_INFO(
              "FaceDetectionWindow",
              std::format("Loading image: {}", fileName.toStdString())
            );
            m_currentImage = cv::imread(fileName.toStdString());

            if (m_currentImage.empty())
            {
                showError("Failed to load image: " + fileName);
                LOG_NAMED_ERROR(
                  "FaceDetectionWindow",
                  std::format(
                    "Failed to load image: {}", fileName.toStdString()
                  )
                );
                return;
            }

            updateImageDisplay(m_currentImage);
            statusBar()->showMessage(
              "Image loaded: " + QFileInfo(fileName).filePath()
            );
        }
    }
}

void FaceDetectionWindow::loadVideoFile(const QString& fileName)
{
    LOG_NAMED_INFO(
      "FaceDetectionWindow",
      std::format("Loading video: {}", fileName.toStdString())
    );

    // Stop any current video processing
    if (m_videoFileActive)
    {
        m_videoCapture.release();
        m_videoFileActive = false;
        if (m_cameraTimer)
        {
            m_cameraTimer->stop();
        }
    }

    // Try to open the video file
    if (!m_videoCapture.open(fileName.toStdString()))
    {
        showError("Failed to open video file: " + fileName);
        LOG_NAMED_ERROR(
          "FaceDetectionWindow",
          std::format("Failed to open video file: {}", fileName.toStdString())
        );
        return;
    }

    // Get video information
    int frameCount =
      static_cast<int>(m_videoCapture.get(cv::CAP_PROP_FRAME_COUNT));
    double fps = m_videoCapture.get(cv::CAP_PROP_FPS);
    int width = static_cast<int>(m_videoCapture.get(cv::CAP_PROP_FRAME_WIDTH));
    int height =
      static_cast<int>(m_videoCapture.get(cv::CAP_PROP_FRAME_HEIGHT));

    m_currentVideoPath = fileName;
    m_videoFileActive = true;

    // Load first frame
    cv::Mat firstFrame;
    if (m_videoCapture.read(firstFrame))
    {
        m_currentImage = firstFrame.clone();
        updateImageDisplay(m_currentImage);

        statusBar()->showMessage(
          QString("Video loaded: %1 (%2x%3, %4 fps, %5 frames)")
            .arg(QFileInfo(fileName).fileName())
            .arg(width)
            .arg(height)
            .arg(QString::number(fps, 'f', 1))
            .arg(frameCount)
        );

        // Reset video to beginning
        m_videoCapture.set(cv::CAP_PROP_POS_FRAMES, 0);

        // Update UI to show video controls
        if (m_startCameraButton && m_stopCameraButton)
        {
            m_startCameraButton->setText("Play Video");
            m_stopCameraButton->setText("Stop Video");
            m_startCameraButton->setEnabled(true);
            m_stopCameraButton->setEnabled(false);
        }

        QMessageBox::information(
          this,
          "Video Loaded",
          QString(
            "Video file loaded successfully!\n\n"
            "Resolution: %1x%2\n"
            "Frame Rate: %3 fps\n"
            "Duration: %4 frames\n\n"
            "Use 'Play Video' button to start playback and face detection."
          )
            .arg(width)
            .arg(height)
            .arg(QString::number(fps, 'f', 1))
            .arg(frameCount)
        );
    }
    else
    {
        showError("Failed to read first frame from video: " + fileName);
        m_videoCapture.release();
        m_videoFileActive = false;
    }
}

void FaceDetectionWindow::startCamera()
{
    // Check if we have a video file loaded instead of camera
    if (m_videoFileActive && m_videoCapture.isOpened())
    {
        // Start video playback
        m_cameraActive = true;

        // Get video FPS and set appropriate timer interval
        double videoFps = m_videoCapture.get(cv::CAP_PROP_FPS);
        int timerInterval = 33; // Default ~30 FPS

        if (videoFps > 0 && videoFps <= 60)
        {
            // Calculate interval based on video FPS, but cap at 60 FPS for
            // performance
            timerInterval = static_cast<int>(1000.0 / std::min(videoFps, 30.0));
        }

        m_cameraTimer->start(timerInterval);

        // Reset performance monitoring for new video session
        if (m_performanceMonitor)
        {
            m_performanceMonitor->resetStatistics();
        }
        m_frameCount = 0;
        m_lastFrameTime = std::chrono::high_resolution_clock::now();

        m_startCameraButton->setEnabled(false);
        m_stopCameraButton->setEnabled(true);
        m_startCameraButton->setText("Play Video");
        m_stopCameraButton->setText("Stop Video");

        statusBar()->showMessage(
          QString("Video playback started (Target FPS: %1)")
            .arg(QString::number(std::min(videoFps, 30.0), 'f', 1))
        );
        return;
    }

    // Normal camera operation
    if (!m_camera.open(0))
    {
        showError(
          "Failed to open camera. Please check if a camera is connected."
        );
        LOG_NAMED_ERROR("FaceDetectionWindow", "Failed to open camera device.");
        return;
    }

    // Set camera properties
    m_camera.set(cv::CAP_PROP_FRAME_WIDTH, 640);
    m_camera.set(cv::CAP_PROP_FRAME_HEIGHT, 480);

    m_cameraActive = true;
    m_cameraTimer->start(30); // 30ms interval (~33 FPS)

    // Reset performance monitoring for new camera session
    if (m_performanceMonitor)
    {
        m_performanceMonitor->resetStatistics();
    }
    m_frameCount = 0;
    m_lastFrameTime = std::chrono::high_resolution_clock::now();

    m_startCameraButton->setEnabled(false);
    m_stopCameraButton->setEnabled(true);

    statusBar()->showMessage("Camera started");
}

void FaceDetectionWindow::stopCamera()
{
    m_cameraActive = false;

    if (m_cameraTimer)
    {
        m_cameraTimer->stop();
    }

    if (m_camera.isOpened())
    {
        m_camera.release();
    }

    // Keep video file loaded but stop playback
    // Don't release m_videoCapture so user can restart playback

    m_startCameraButton->setEnabled(true);
    m_stopCameraButton->setEnabled(false);

    if (m_videoFileActive)
    {
        m_startCameraButton->setText("Play Video");
        m_stopCameraButton->setText("Stop Video");
        statusBar()->showMessage("Video playback stopped");
    }
    else
    {
        m_startCameraButton->setText("Start Camera");
        m_stopCameraButton->setText("Stop Camera");
        statusBar()->showMessage("Camera stopped");
    }
}

void FaceDetectionWindow::processFrame()
{
    // Early return if not active
    if (!m_cameraActive)
    {
        return;
    }

    // Record frame processing start time
    auto frameStartTime = std::chrono::high_resolution_clock::now();

    cv::Mat frame;
    bool frameRead = false;
    size_t detectionCount = 0;
    float avgConfidence = 0.0f;

    // Choose source: video file or camera
    if (m_videoFileActive && m_videoCapture.isOpened())
    {
        frameRead = m_videoCapture.read(frame);

        // Check if video has ended
        if (!frameRead)
        {
            // Reset to beginning for looping
            m_videoCapture.set(cv::CAP_PROP_POS_FRAMES, 0);
            frameRead = m_videoCapture.read(frame);

            if (!frameRead)
            {
                // Video file issue, stop playback
                stopCamera();
                statusBar()->showMessage("Video playback ended");
                return;
            }
        }
    }
    else if (m_camera.isOpened())
    {
        frameRead = m_camera.read(frame);
    }
    else
    {
        return;
    }

    if (frameRead && !frame.empty())
    {
        m_currentImage = frame.clone();

        // Record face detection start time
        auto detectionStartTime = std::chrono::high_resolution_clock::now();

        // Automatically detect faces in camera/video mode
        cv::Mat processedFrame = detectFacesInImage(frame);

        // Calculate detection metrics if we have a detector
        if (m_faceDetector && m_faceDetector->isLoaded())
        {
            std::vector<DetectionResult> detections =
              m_faceDetector->detectFaces(frame);
            detectionCount = detections.size();

            // Calculate average confidence
            if (!detections.empty())
            {
                float totalConfidence = 0.0f;
                for (const auto& detection : detections)
                {
                    totalConfidence += detection.confidence;
                }
                avgConfidence = totalConfidence / detections.size();
            }
        }

        updateImageDisplay(processedFrame);

        // Record frame processing end time and track performance
        auto frameEndTime = std::chrono::high_resolution_clock::now();
        auto processingTime =
          std::chrono::duration_cast<std::chrono::milliseconds>(
            frameEndTime - detectionStartTime
          );

        // Track frame result in performance monitor
        if (m_performanceMonitor)
        {
            m_performanceMonitor->trackFrameResult(
              processingTime,
              detectionCount,
              avgConfidence,
              false, // frameSkipped
              false, // frameDropped
              false  // processingFailed
            );
        }

        // Update frame count and timing for FPS calculation
        m_frameCount++;

        // Update status for video files
        if (m_videoFileActive)
        {
            // Get current frame position for video status
            int currentFrame =
              static_cast<int>(m_videoCapture.get(cv::CAP_PROP_POS_FRAMES));
            int totalFrames =
              static_cast<int>(m_videoCapture.get(cv::CAP_PROP_FRAME_COUNT));
            double videoFps = m_videoCapture.get(cv::CAP_PROP_FPS);

            QString videoStatus =
              QString("Video: Frame %1/%2 (FPS: %3) - %4 face(s) detected")
                .arg(currentFrame)
                .arg(totalFrames)
                .arg(QString::number(videoFps, 'f', 1))
                .arg(detectionCount);

            statusBar()->showMessage(videoStatus);
        }

        // Force update performance display to ensure real-time updates
        updatePerformanceDisplay();
    }
}

void FaceDetectionWindow::detectFaces()
{
    if (m_currentImage.empty())
    {
        showError(
          "No image loaded. Please load an image or start the camera first."
        );
        LOG_NAMED_WARNING("FaceDetectionWindow", "No image loaded.");
        return;
    }

    if (!m_faceDetector || !m_faceDetector->isLoaded())
    {
        showError("No face detection model loaded. Please load a model first.");
        LOG_NAMED_WARNING("FaceDetectionWindow", "No model loaded.");
        return;
    }

    m_progressBar->setVisible(true);
    m_progressBar->setRange(0, 0); // Indeterminate progress

    cv::Mat result = detectFacesInImage(m_currentImage);
    updateImageDisplay(result);

    m_progressBar->setVisible(false);
}

cv::Mat FaceDetectionWindow::detectFacesInImage(const cv::Mat& image)
{
    if (image.empty() || !m_faceDetector || !m_faceDetector->isLoaded())
    {
        return image;
    }

    cv::Mat result = image.clone();

    // Detect faces using the current detector
    std::vector<DetectionResult> detections =
      m_faceDetector->detectFaces(image);

    // Draw rectangles around detected faces
    for (const auto& detection : detections)
    {
        const cv::Rect& face = detection.boundingBox;
        float confidence = detection.confidence;

        // Choose color based on confidence (green for high, yellow for medium,
        // red for low)
        cv::Scalar color;
        if (confidence >= 0.8f)
        {
            color = cv::Scalar(0, 255, 0); // Green
        }
        else if (confidence >= 0.6f)
        {
            color = cv::Scalar(0, 255, 255); // Yellow
        }
        else
        {
            color = cv::Scalar(0, 165, 255); // Orange
        }

        cv::rectangle(result, face, color, 2);

        // Add label with confidence score
        std::string label = "Face";
        if (confidence < 1.0f)
        { // Only show confidence if it's meaningful (not cascade default)
            label +=
              " (" + std::to_string(static_cast<int>(confidence * 100)) + "%)";
        }

        int baseline = 0;
        cv::Size labelSize =
          cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseline);

        // Draw label background
        cv::rectangle(
          result,
          cv::Point(face.x, face.y - labelSize.height - 10),
          cv::Point(face.x + labelSize.width, face.y),
          color,
          -1
        );

        cv::putText(
          result,
          label,
          cv::Point(face.x, face.y - 5),
          cv::FONT_HERSHEY_SIMPLEX,
          0.5,
          cv::Scalar(0, 0, 0),
          1
        );
    }

    // Update status with detector information
    QString detectorName =
      QString::fromStdString(m_faceDetector->getModelInfo());
    QString statusText = QString("Status: %1 - Detected %2 face(s)")
                           .arg(detectorName.split(" - ").last())
                           .arg(detections.size());
    LOG_NAMED_INFO(
      "FaceDetectionWindow",
      std::format(
        "{} detected {} face(s).", detectorName.toStdString(), detections.size()
      )
    );

    statusBar()->showMessage(QString("Found %1 face(s) using %2")
                               .arg(detections.size())
                               .arg(detectorName.split(" - ").first()));

    return result;
}

QImage FaceDetectionWindow::matToQImage(const cv::Mat& mat)
{
    if (mat.empty())
    {
        return QImage();
    }

    switch (mat.type())
    {
    case CV_8UC4: {
        QImage qimg(
          mat.data, mat.cols, mat.rows, mat.step, QImage::Format_ARGB32
        );
        return qimg.rgbSwapped();
    }
    case CV_8UC3: {
        QImage qimg(
          mat.data, mat.cols, mat.rows, mat.step, QImage::Format_RGB888
        );
        return qimg.rgbSwapped();
    }
    case CV_8UC1: {
        QImage qimg(
          mat.data, mat.cols, mat.rows, mat.step, QImage::Format_Grayscale8
        );
        return qimg;
    }
    default:
        break;
    }

    return QImage();
}

void FaceDetectionWindow::updateImageDisplay(const cv::Mat& image)
{
    if (image.empty())
    {
        return;
    }

    QImage qimg = matToQImage(image);
    if (qimg.isNull())
    {
        return;
    }

    // Scale image to fit label while maintaining aspect ratio
    QPixmap pixmap = QPixmap::fromImage(qimg);
    QSize labelSize = m_imageLabel->size();
    pixmap =
      pixmap.scaled(labelSize, Qt::KeepAspectRatio, Qt::SmoothTransformation);

    m_imageLabel->setPixmap(pixmap);
}

void FaceDetectionWindow::showError(const QString& message)
{
    QMessageBox::critical(this, "Error", message);
    LOG_NAMED_ERROR("FaceDetectionWindow", message.toStdString());

    statusBar()->showMessage("Error: " + message);
}

void FaceDetectionWindow::updateModelInfo()
{
    if (!m_faceDetector || !m_faceDetector->isLoaded())
    {
        m_modelInfoLabel->setText("Model: None loaded");
        LOG_NAMED_WARNING("FaceDetectionWindow", "No model loaded.");
        return;
    }

    QString modelInfo = QString::fromStdString(m_faceDetector->getModelInfo());
    QFileInfo fileInfo(m_modelPath);

    QString detectorType;
    switch (m_currentDetectorType)
    {
    case FaceDetectorFactory::DetectorType::CASCADE:
        detectorType = "Haar Cascade";
        break;
    case FaceDetectorFactory::DetectorType::DNN:
        detectorType = "Deep Neural Network";
        break;
    default:
        detectorType = "Unknown";
        break;
    }

    m_modelInfoLabel->setText(QString("Model: %1 (%2) - %3")
                                .arg(fileInfo.fileName())
                                .arg(detectorType)
                                .arg(fileInfo.suffix().toUpper()));
}

QString FaceDetectionWindow::getModelFilterString() const
{
    auto supportedExtensions = FaceDetectorFactory::getAllSupportedExtensions();

    QStringList filterParts;

    // Add specific format filters
    filterParts << "Haar Cascade Files (*.xml *.cascade)";
    filterParts << "ONNX Models (*.onnx)";
    filterParts << "TensorFlow Models (*.pb)";
    filterParts << "Caffe Models (*.caffemodel)";

    // Create all supported files filter
    QStringList allExtensions;
    for (const auto& ext : supportedExtensions)
    {
        allExtensions << "*" + QString::fromStdString(ext);
    }
    filterParts.prepend(
      "All Supported Models (" + allExtensions.join(" ") + ")"
    );

    // Add all files filter
    filterParts << "All Files (*)";

    return filterParts.join(";;");
}

void FaceDetectionWindow::tryLoadDefaultModel()
{
    LOG_NAMED_INFO(
      "FaceDetectionWindow", "Attempting to load default model..."
    );

    // Try to find and load a default model
    std::string defaultModelPath =
      ModelValidator::getDefaultModelPath(true); // Prefer DNN models

    if (defaultModelPath.empty())
    {
        LOG_NAMED_INFO(
          "FaceDetectionWindow", "No DNN model found, trying cascade models..."
        );
        // Try without DNN preference
        defaultModelPath = ModelValidator::getDefaultModelPath(false);
    }

    if (!defaultModelPath.empty())
    {
        LOG_NAMED_INFO(
          "FaceDetectionWindow",
          std::format("Found potential default model: {}", defaultModelPath)
        );

        // Validate the default model
        ValidationResult validation =
          ModelValidator::validateModel(defaultModelPath);
        if (validation.isValid)
        {
            LOG_NAMED_INFO(
              "FaceDetectionWindow",
              "Model validation successful, creating detector..."
            );

            // Try to create and load the detector
            auto detector =
              FaceDetectorFactory::createOptimizedDetector(defaultModelPath);
            if (detector)
            {
                m_faceDetector = std::move(detector);
                m_modelPath = QString::fromStdString(defaultModelPath);
                m_currentDetectorType =
                  FaceDetectorFactory::detectModelType(defaultModelPath);

                updateModelInfo();

                statusBar()->showMessage(
                  "Default model loaded automatically: " +
                  QFileInfo(m_modelPath).fileName()
                );

                LOG_NAMED_INFO(
                  "FaceDetectionWindow",
                  std::format(
                    "Successfully loaded default model: {}", defaultModelPath
                  )
                );

                // Show warning if there was one during validation
                if (!validation.warningMessage.empty())
                {
                    // Don't show popup for auto-loaded models, just log to
                    // status
                    statusBar()->showMessage(
                      statusBar()->currentMessage() + " (Warning: " +
                      QString::fromStdString(validation.warningMessage) + ")"
                    );
                }
                return;
            }
            else
            {
                LOG_NAMED_ERROR(
                  "FaceDetectionWindow",
                  std::format(
                    "Failed to create detector for default model: {}",
                    defaultModelPath
                  )
                );
            }
        }
        else
        {
            LOG_NAMED_ERROR(
              "FaceDetectionWindow",
              std::format(
                "Default model validation failed: {} - {}",
                defaultModelPath,
                validation.errorMessage
              )
            );
        }
    }
    else
    {
        LOG_NAMED_WARNING(
          "FaceDetectionWindow", "No default model found in any search paths"
        );
    }

    // If we get here, no default model could be loaded
    statusBar()->showMessage(
      "No default model available - Load a model to get started"
    );
    LOG_NAMED_INFO("FaceDetectionWindow", "No default model could be loaded");
}

void FaceDetectionWindow::onCudaToggled(bool enabled)
{
    if (!m_faceDetector)
    {
        return;
    }

    bool success = false;

    // Handle DNN detector (YunetFaceDetector)
    if (m_currentDetectorType == FaceDetectorFactory::DetectorType::DNN)
    {
        auto* dnnDetector =
          dynamic_cast<YunetFaceDetector*>(m_faceDetector.get());
        if (dnnDetector)
        {
            success = dnnDetector->setCudaEnabled(enabled);
        }
    }
    // Handle Cascade detector (CascadeFaceDetector)
    else if (m_currentDetectorType ==
             FaceDetectorFactory::DetectorType::CASCADE)
    {
        auto* cascadeDetector =
          dynamic_cast<CascadeFaceDetector*>(m_faceDetector.get());
        if (cascadeDetector)
        {
            success = cascadeDetector->setCudaEnabled(enabled);
        }
    }

    if (!success && enabled)
    {
        // CUDA enable failed, revert checkbox
        m_cudaCheckBox->setChecked(false);
        QMessageBox::warning(
          this,
          "CUDA Error",
          "Failed to enable CUDA acceleration. Using CPU fallback."
        );
    }

    // Update model info to reflect CUDA status
    updateModelInfo();

    // Show status message
    if (success)
    {
        statusBar()->showMessage(
          enabled ? "CUDA acceleration enabled" : "CUDA acceleration disabled"
        );
    }
}

void FaceDetectionWindow::toggleVideoEnhancement()
{
    if (!m_faceDetector)
    {
        QMessageBox::warning(
          this, "No Model", "Please load a face detection model first."
        );
        return;
    }

    if (!m_videoProcessingActive)
    {
        // TODO: Implement video enhancement logic, for now just use the
        // existing detector pattern Start enhanced video processing
        if (!m_videoDetector)
        {
            // Create shared pointer from unique pointer for VideoFaceDetector
            std::shared_ptr<IFaceDetector> sharedDetector;
            if (m_faceDetector)
            {
                // Note: In a full implementation, we'd need a way to share the
                // detector. For now, we'll use the existing detector pattern
                sharedDetector = std::static_pointer_cast<IFaceDetector>(
                  std::shared_ptr<IFaceDetector>(
                    m_faceDetector.get(), [](IFaceDetector*) {}
                  )
                );
            }

            m_videoDetector =
              std::make_unique<VideoFaceDetector>(sharedDetector);

            // Configure for optimal performance
            VideoFaceDetector::ProcessingConfig config =
              VideoFaceDetector::getRecommendedConfig(true); // prioritize speed

            if (m_cudaCheckBox->isChecked())
            {
                config.gpuAcceleration = true;
            }

            if (!m_videoDetector->initialize(config))
            {
                QMessageBox::warning(
                  this,
                  "Initialization Error",
                  "Failed to initialize video processing system."
                );
                m_videoDetector.reset();
                return;
            }
        }

        m_videoProcessingActive = true;
        m_videoProcessingButton->setText("Stop Enhanced");
        m_videoProcessingButton->setStyleSheet(
          "QPushButton { background-color: #ff6b6b; }"
        );
        statusBar()->showMessage("Enhanced video processing started");
    }
    else
    {
        // Stop enhanced video processing
        if (m_videoDetector)
        {
            m_videoDetector->stopProcessing();
        }

        m_videoProcessingActive = false;
        m_videoProcessingButton->setText("Enhanced Video");
        m_videoProcessingButton->setStyleSheet("");
        statusBar()->showMessage("Enhanced video processing stopped");
    }
}

void FaceDetectionWindow::showPerformanceDialog()
{
    if (!m_performanceMonitor)
    {
        QMessageBox::information(
          this,
          "Performance Monitor",
          "Performance monitoring is not available.\n\nThe performance monitor "
          "requires initialization during application startup."
        );
        return;
    }

    QString report;
    try
    {
        // Safely get performance metrics with timeout protection
        auto stats = m_performanceMonitor->getCurrentMetrics();

        // Get additional information for video processing
        QString sourceInfo = "Unknown";
        QString sourceDetails = "";

        if (m_videoFileActive && m_videoCapture.isOpened())
        {
            sourceInfo = "Video File";
            int currentFrame =
              static_cast<int>(m_videoCapture.get(cv::CAP_PROP_POS_FRAMES));
            int totalFrames =
              static_cast<int>(m_videoCapture.get(cv::CAP_PROP_FRAME_COUNT));
            double videoFps = m_videoCapture.get(cv::CAP_PROP_FPS);
            int width =
              static_cast<int>(m_videoCapture.get(cv::CAP_PROP_FRAME_WIDTH));
            int height =
              static_cast<int>(m_videoCapture.get(cv::CAP_PROP_FRAME_HEIGHT));

            sourceDetails =
              QString(
                "Video: %1\nFrame: %2/%3\nVideo FPS: %4\nResolution: %5x%6"
              )
                .arg(QFileInfo(m_currentVideoPath).fileName())
                .arg(currentFrame)
                .arg(totalFrames)
                .arg(QString::number(videoFps, 'f', 1))
                .arg(width)
                .arg(height);
        }
        else if (m_cameraActive && m_camera.isOpened())
        {
            sourceInfo = "Camera";
            int width =
              static_cast<int>(m_camera.get(cv::CAP_PROP_FRAME_WIDTH));
            int height =
              static_cast<int>(m_camera.get(cv::CAP_PROP_FRAME_HEIGHT));
            sourceDetails = QString("Resolution: %1x%2").arg(width).arg(height);
        }
        else
        {
            sourceInfo = "Static Image";
            if (!m_currentImage.empty())
            {
                sourceDetails = QString("Resolution: %1x%2")
                                  .arg(m_currentImage.cols)
                                  .arg(m_currentImage.rows);
            }
        }

        // Create a comprehensive report
        report =
          QString(
            "=== Performance Monitor Report ===\n\n"
            "=== Source Information ===\n"
            "Source Type: %1\n"
            "%2\n\n"
            "=== Real-time Performance ===\n"
            "Current FPS: %3\n"
            "Average Frame Time: %4 ms\n"
            "Peak Frame Time: %5 ms\n"
            "Frames Processed: %6\n"
            "Skipped Frames: %7\n"
            "Performance Health: %8%%\n\n"
            "=== Detection Statistics ===\n"
            "Total Faces Detected: %9\n"
            "Average Faces per Frame: %10\n"
            "Average Confidence: %11%%\n\n"
            "=== System Status ===\n"
            "GPU Memory Used: %12 MB\n"
            "GPU Acceleration: %13\n"
            "Processing Resolution: %14x%15\n"
            "Frame Skip Level: %16\n\n"
            "=== Model Information ===\n"
            "Detector Type: %17\n"
            "Model Path: %18\n\n"
            "Note: Performance metrics update in real-time during processing."
          )
            .arg(sourceInfo)
            .arg(
              sourceDetails.isEmpty() ? "No additional details" : sourceDetails
            )
            .arg(QString::number(stats.currentFPS, 'f', 1))
            .arg(QString::number(stats.avgFrameTime.count(), 'f', 2))
            .arg(QString::number(stats.maxFrameTime.count(), 'f', 2))
            .arg(stats.processedFrames)
            .arg(stats.skippedFrames)
            .arg(
              QString::number(
                m_performanceMonitor->getPerformanceHealthScore() * 100, 'f', 1
              )
            )
            .arg(stats.totalDetections)
            .arg(QString::number(stats.avgDetectionsPerFrame, 'f', 2))
            .arg(QString::number(stats.avgConfidence * 100, 'f', 1))
            .arg(
              QString::number(stats.gpuMemoryUsed / (1024.0 * 1024.0), 'f', 1)
            )
            .arg(
              m_cudaCheckBox && m_cudaCheckBox->isChecked() ? "Enabled"
                                                            : "Disabled"
            )
            .arg(stats.currentResolution.width)
            .arg(stats.currentResolution.height)
            .arg(stats.currentFrameSkip)
            .arg(
              m_faceDetector
                ? QString::fromStdString(m_faceDetector->getModelInfo())
                    .split(" - ")
                    .first()
                : "None"
            )
            .arg(
              m_modelPath.isEmpty() ? "None loaded"
                                    : QFileInfo(m_modelPath).fileName()
            );
    }
    catch (const std::exception& e)
    {
        report =
          QString(
            "Error generating performance report: %1\n\n"
            "This may indicate the performance monitor is not properly "
            "initialized or there are no performance metrics available yet.\n\n"
            "Try processing some images or video first."
          )
            .arg(e.what());
    }
    catch (...)
    {
        report =
          "Unknown error occurred while generating performance report.\n\n"
          "Try restarting the application if this persists.";
    }

    QDialog* dialog = new QDialog(this);
    dialog->setWindowTitle("Performance Monitor");
    dialog->setMinimumSize(700, 500);
    dialog->setAttribute(Qt::WA_DeleteOnClose); // Auto-cleanup

    QVBoxLayout* layout = new QVBoxLayout(dialog);

    QTextEdit* textEdit = new QTextEdit(dialog);
    textEdit->setPlainText(report);
    textEdit->setReadOnly(true);
    textEdit->setFont(QFont("Courier", 9));

    // Add refresh button
    QHBoxLayout* buttonLayout = new QHBoxLayout();
    QPushButton* refreshButton = new QPushButton("Refresh", dialog);
    QPushButton* closeButton = new QPushButton("Close", dialog);

    buttonLayout->addWidget(refreshButton);
    buttonLayout->addStretch();
    buttonLayout->addWidget(closeButton);

    connect(
      refreshButton,
      &QPushButton::clicked,
      [this, textEdit]()
      {
          // Refresh the performance data
          if (m_performanceMonitor)
          {
              try
              {
                  auto stats = m_performanceMonitor->getCurrentMetrics();
                  QString updatedReport =
                    QString(
                      "=== Performance Monitor Report (Refreshed) ===\n\n"
                      "Current FPS: %1\n"
                      "Average Frame Time: %2 ms\n"
                      "Peak Frame Time: %3 ms\n"
                      "Frames Processed: %4\n"
                      "Skipped Frames: %5\n"
                      "Performance Health: %6%%\n"
                      "Total Detections: %7\n"
                      "Average Detections per Frame: %8\n\n"
                      "Last Updated: %9\n"
                    )
                      .arg(QString::number(stats.currentFPS, 'f', 1))
                      .arg(QString::number(stats.avgFrameTime.count(), 'f', 2))
                      .arg(QString::number(stats.maxFrameTime.count(), 'f', 2))
                      .arg(stats.processedFrames)
                      .arg(stats.skippedFrames)
                      .arg(
                        QString::number(
                          m_performanceMonitor->getPerformanceHealthScore() *
                            100,
                          'f',
                          1
                        )
                      )
                      .arg(stats.totalDetections)
                      .arg(QString::number(stats.avgDetectionsPerFrame, 'f', 2))
                      .arg(QDateTime::currentDateTime().toString());

                  textEdit->setPlainText(updatedReport);
              }
              catch (...)
              {
                  textEdit->setPlainText("Error refreshing performance data.");
              }
          }
      }
    );

    connect(closeButton, &QPushButton::clicked, dialog, &QDialog::accept);

    layout->addWidget(textEdit);
    layout->addLayout(buttonLayout);

    dialog->show(); // Use show() instead of exec() to prevent blocking
}

void FaceDetectionWindow::showConfigDialog()
{
    if (!m_configManager)
    {
        QMessageBox::information(
          this,
          "Configuration Manager",
          "Configuration management is not available."
        );
        return;
    }

    QDialog* dialog = new QDialog(this);
    dialog->setWindowTitle("Configuration Manager");
    dialog->setMinimumSize(500, 350);

    QVBoxLayout* layout = new QVBoxLayout(dialog);

    // Profile selection
    QGroupBox* profileGroup = new QGroupBox("Detection Profiles", dialog);
    QVBoxLayout* profileLayout = new QVBoxLayout(profileGroup);

    QComboBox* profileCombo = new QComboBox(dialog);
    auto profiles = m_configManager->getAvailableProfiles();
    for (const auto& profile : profiles)
    {
        profileCombo->addItem(QString::fromStdString(profile));
    }

    QPushButton* loadProfileButton = new QPushButton("Load Profile", dialog);
    QPushButton* saveProfileButton =
      new QPushButton("Save Current as...", dialog);

    profileLayout->addWidget(new QLabel("Available Profiles:", dialog));
    profileLayout->addWidget(profileCombo);

    QHBoxLayout* profileButtonLayout = new QHBoxLayout();
    profileButtonLayout->addWidget(loadProfileButton);
    profileButtonLayout->addWidget(saveProfileButton);
    profileLayout->addLayout(profileButtonLayout);

    // Hardware info
    QGroupBox* hardwareGroup = new QGroupBox("Hardware Information", dialog);
    QVBoxLayout* hardwareLayout = new QVBoxLayout(hardwareGroup);

    auto hwInfo = ConfigurationManager::detectHardwareCapabilities();
    QString hwText =
      QString("GPU Available: %1\nCUDA Support: %2\nCPU Cores: %3")
        .arg(hwInfo.hasGpu ? "Yes" : "No")
        .arg(hwInfo.hasOpenCVCuda ? "Yes" : "No")
        .arg(hwInfo.cpuCores);

    QLabel* hwLabel = new QLabel(hwText, dialog);
    hardwareLayout->addWidget(hwLabel);

    // Close button
    QPushButton* closeButton = new QPushButton("Close", dialog);
    connect(closeButton, &QPushButton::clicked, dialog, &QDialog::accept);

    layout->addWidget(profileGroup);
    layout->addWidget(hardwareGroup);
    layout->addWidget(closeButton);

    // Connect profile buttons
    connect(
      loadProfileButton,
      &QPushButton::clicked,
      [this, profileCombo]()
      {
          QString profileName = profileCombo->currentText();
          if (!profileName.isEmpty())
          {
              if (m_configManager->setCurrentProfile(profileName.toStdString()))
              {
                  statusBar()->showMessage("Loaded profile: " + profileName);
              }
              else
              {
                  QMessageBox::warning(
                    this, "Load Error", "Failed to load profile: " + profileName
                  );
              }
          }
      }
    );

    connect(
      saveProfileButton,
      &QPushButton::clicked,
      [this]()
      {
          bool ok;
          QString name = QInputDialog::getText(
            this, "Save Profile", "Profile name:", QLineEdit::Normal, "", &ok
          );
          if (ok && !name.isEmpty())
          {
              auto profile = m_configManager->createProfileFromCurrentSettings(
                name.toStdString()
              );
              if (m_configManager->saveDetectionProfile(
                    name.toStdString(), profile
                  ))
              {
                  statusBar()->showMessage("Saved profile: " + name);
              }
              else
              {
                  QMessageBox::warning(
                    this, "Save Error", "Failed to save profile: " + name
                  );
              }
          }
      }
    );

    dialog->exec();
    dialog->deleteLater();
}

void FaceDetectionWindow::updatePerformanceDisplay()
{
    if (!m_performanceMonitor)
    {
        return;
    }

    auto stats = m_performanceMonitor->getCurrentMetrics();

    // Calculate real-time FPS based on actual frame processing
    auto currentTime = std::chrono::high_resolution_clock::now();
    auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
      currentTime - m_lastFrameTime
    );

    double realTimeFPS = 0.0;
    if (elapsed.count() > 0 && m_cameraActive)
    {
        // Calculate FPS from actual frame interval
        realTimeFPS = 1000.0 / elapsed.count();

        // Use the higher of performance monitor FPS or real-time calculation
        realTimeFPS =
          std::max(realTimeFPS, static_cast<double>(stats.currentFPS));
    }
    else
    {
        realTimeFPS = stats.currentFPS;
    }

    // Update FPS display with real-time calculation
    m_fpsLabel->setText(QString("FPS: %1").arg(realTimeFPS, 0, 'f', 1));

    // Update performance status with color coding
    float healthScore = m_performanceMonitor->getPerformanceHealthScore();
    QString status;
    QString color;

    if (healthScore > 0.8f)
    {
        status = "Excellent";
        color = "#006600"; // Green
    }
    else if (healthScore > 0.6f)
    {
        status = "Good";
        color = "#0066CC"; // Blue
    }
    else if (healthScore > 0.4f)
    {
        status = "Fair";
        color = "#FF6600"; // Orange
    }
    else
    {
        status = "Poor";
        color = "#CC0000"; // Red
    }

    // Show different status for video vs camera
    QString performanceText;
    if (m_videoFileActive)
    {
        performanceText = QString("Video Performance: %1").arg(status);
    }
    else if (m_cameraActive)
    {
        performanceText = QString("Camera Performance: %1").arg(status);
    }
    else
    {
        performanceText = QString("Performance: %1").arg(status);
    }

    m_performanceLabel->setText(performanceText);
    m_performanceLabel->setStyleSheet(
      QString("QLabel { font-weight: bold; color: %1; }").arg(color)
    );

    // Update timing for next calculation
    if (m_cameraActive)
    {
        m_lastFrameTime = currentTime;
    }

    // Reset frame count periodically for accurate FPS calculation
    auto timeSinceStart = std::chrono::duration_cast<std::chrono::seconds>(
      currentTime - m_lastFrameTime
    );

    if (timeSinceStart.count() >= 5) // Reset every 5 seconds
    {
        m_frameCount = 0;
        m_lastFrameTime = currentTime;
    }
}

void FaceDetectionWindow::onFrameProcessed(
  const VideoFaceDetector::FrameResult& result
)
{
    // Update the displayed image
    if (!result.processedFrame.empty())
    {
        updateImageDisplay(result.processedFrame);
    }

    // Track performance
    if (m_performanceMonitor)
    {
        m_performanceMonitor->trackFrameResult(
          result.processingTime,
          result.detections.size(),
          result.detections.empty()
            ? 0.0f
            : std::accumulate(
                result.detections.begin(),
                result.detections.end(),
                0.0f,
                [](float sum, const DetectionResult& det)
                {
                    return sum + det.confidence;
                }
              ) /
                result.detections.size(),
          false, // frameSkipped
          false, // frameDropped
          false  // processingFailed
        );
    }

    m_frameCount++;

    // Update status bar with detection info
    statusBar()->showMessage(
      QString("Processed frame %1: %2 faces detected in %3ms")
        .arg(result.frameIndex)
        .arg(result.detections.size())
        .arg(result.processingTime.count())
    );
}
