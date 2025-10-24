// Copyright (c) 2025 JackLee
//
// This software is released under the MIT License.
// https://opensource.org/licenses/MIT

#ifndef YUNETFACEDETECTOR_H
#define YUNETFACEDETECTOR_H

#include "IFaceDetector.h"
#include <opencv2/dnn.hpp>
#include <opencv2/objdetect.hpp>
#include <opencv2/imgproc.hpp>
#ifdef OPENCV_DNN_CUDA
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudaarithm.hpp>
#endif

/**
 * @brief Face detector implementation using OpenCV DNN module
 *
 * This implementation uses deep neural networks for face detection, providing
 * superior accuracy and robustness compared to traditional methods. It supports
 * ONNX models such as YuNet which offer excellent performance across various
 * lighting conditions and face orientations.
 *
 * Supported formats: ONNX (.onnx), TensorFlow (.pb), Caffe (.caffemodel), etc.
 *
 * Model Specifications:
 * - Input size: 320x320 (can be resized)
 * - Output: Detections with confidence scores and landmarks
 * - Optimized for real-time face detection
 */
class YunetFaceDetector : public IFaceDetector
{
  public:
    YunetFaceDetector();
    ~YunetFaceDetector() override = default;

    // Disable copy constructor and assignment operator for resource management
    YunetFaceDetector(const YunetFaceDetector&) = delete;
    YunetFaceDetector& operator=(const YunetFaceDetector&) = delete;

    // Enable move semantics
    YunetFaceDetector(YunetFaceDetector&&) = default;
    YunetFaceDetector& operator=(YunetFaceDetector&&) = default;

    /**
     * @brief Load DNN model from file
     * @param modelPath Path to the model file (ONNX, TensorFlow, etc.)
     * @return True if model loaded successfully, false otherwise
     */
    bool loadModel(const std::string& modelPath) override;

    /**
     * @brief Detect faces using DNN model
     * @param image Input image (BGR format preferred)
     * @return Vector of detection results with bounding boxes and confidence
     * scores
     */
    std::vector<DetectionResult> detectFaces(const cv::Mat& image) override;

    /**
     * @brief Check if DNN model is loaded
     * @return True if model is loaded and ready for detection
     */
    bool isLoaded() const override;

    /**
     * @brief Get information about the loaded DNN model
     * @return String describing the model status and file
     */
    std::string getModelInfo() const override;

    /**
     * @brief Get supported file extensions for DNN models
     * @return Vector of supported extensions
     */
    std::vector<std::string> getSupportedExtensions() const override;

    /**
     * @brief Configure DNN detection parameters
     * @param scaleFactor Not used for DNN (kept for interface compatibility)
     * @param minNeighbors Not used for DNN (kept for interface compatibility)
     * @param minSize Minimum detection size
     * @param maxSize Maximum detection size
     */
    void setDetectionParams(
      double scaleFactor = 1.1,
      int minNeighbors = 3,
      const cv::Size& minSize = cv::Size(30, 30),
      const cv::Size& maxSize = cv::Size()
    ) override;

    /**
     * @brief Set confidence threshold for detections
     * @param threshold Minimum confidence score (0.0 to 1.0)
     */
    void setConfidenceThreshold(float threshold);

    /**
     * @brief Set Non-Maximum Suppression threshold
     * @param threshold NMS threshold for overlapping detection removal
     */
    void setNmsThreshold(float threshold);

    /**
     * @brief Set input size for the DNN model
     * @param size Input size (width, height) - should match model requirements
     */
    void setInputSize(const cv::Size& size);

    /**
     * @brief Enable or disable CUDA acceleration if available
     * @param enable True to enable CUDA, false to use CPU
     * @return True if CUDA was successfully enabled/disabled, false if CUDA is
     * not available
     */
    bool setCudaEnabled(bool enable);

    /**
     * @brief Check if CUDA acceleration is currently enabled
     * @return True if CUDA is enabled and available, false otherwise
     */
    bool isCudaEnabled() const;

    /**
     * @brief Check if CUDA is available on the system
     * @return True if CUDA devices are available, false otherwise
     */
    static bool isCudaAvailable();

  private:
    cv::Ptr<cv::FaceDetectorYN> m_detector;
    std::string m_modelPath;

    // DNN-specific parameters
    cv::Size m_inputSize = cv::Size(320, 320);
    float m_confidenceThreshold = 0.6f;
    float m_nmsThreshold = 0.3f;
    cv::Size m_minSize = cv::Size(30, 30);
    cv::Size m_maxSize = cv::Size();

    // CUDA acceleration settings
    bool m_cudaEnabled = false;
    bool m_cudaAvailable = false;

    /**
     * @brief Convert FaceDetectorYN output to our DetectionResult format
     * @param faces Output from FaceDetectorYN::detect
     * @param imageSize Original image size
     * @return Vector of detection results
     */
    std::vector<DetectionResult> convertOutput(
      const cv::Mat& faces, const cv::Size& imageSize
    ) const;

    /**
     * @brief Determine model type from file extension
     * @param modelPath Path to model file
     * @return True if the model type is supported
     */
    bool isSupportedModelType(const std::string& modelPath) const;

    /**
     * @brief Initialize CUDA availability detection
     */
    void initializeCuda();

    /**
     * @brief Recreate detector with current settings
     * @return True if detector was successfully recreated
     */
    bool recreateDetector();
};

#endif // YUNETFACEDETECTOR_H
