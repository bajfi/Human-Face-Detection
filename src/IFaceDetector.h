// Copyright (c) 2025 JackLee
//
// This software is released under the MIT License.
// https://opensource.org/licenses/MIT

#ifndef IFACEDETECTOR_H
#define IFACEDETECTOR_H

#include <opencv2/opencv.hpp>
#include <vector>
#include <string>
#include <memory>

/**
 * @brief Detection result structure containing bounding box and confidence
 */
struct DetectionResult
{
    cv::Rect boundingBox;
    float confidence;

    DetectionResult(const cv::Rect& box, float conf = 1.0f)
      : boundingBox(box), confidence(conf)
    {}
};

/**
 * @brief Abstract interface for face detection implementations
 *
 * This interface provides a unified API for different face detection backends,
 * enabling seamless switching between Haar Cascades, DNN models, and future
 * implementations.
 *
 * Design Principles:
 * - Interface Segregation: Clean, minimal interface focused on face detection
 * - Dependency Inversion: Depend on abstractions, not concrete implementations
 * - Single Responsibility: Each detector handles one specific model type
 */
class IFaceDetector
{
  public:
    virtual ~IFaceDetector() = default;

    /**
     * @brief Load model from file
     * @param modelPath Path to the model file
     * @return True if model loaded successfully, false otherwise
     */
    virtual bool loadModel(const std::string& modelPath) = 0;

    /**
     * @brief Detect faces in an image
     * @param image Input image (BGR or grayscale)
     * @return Vector of detection results with bounding boxes and confidence
     * scores
     */
    virtual std::vector<DetectionResult> detectFaces(const cv::Mat& image) = 0;

    /**
     * @brief Check if model is loaded and ready for detection
     * @return True if model is loaded, false otherwise
     */
    virtual bool isLoaded() const = 0;

    /**
     * @brief Get model information/description
     * @return String describing the model type and status
     */
    virtual std::string getModelInfo() const = 0;

    /**
     * @brief Get supported file extensions for this detector
     * @return Vector of file extensions (e.g., {".xml", ".cascade"})
     */
    virtual std::vector<std::string> getSupportedExtensions() const = 0;

    /**
     * @brief Configure detection parameters
     * @param scaleFactor Scale factor for multi-scale detection
     * @param minNeighbors Minimum neighbors for detection validation
     * @param minSize Minimum detection size
     * @param maxSize Maximum detection size (empty for no limit)
     */
    virtual void setDetectionParams(
      double scaleFactor = 1.1,
      int minNeighbors = 3,
      const cv::Size& minSize = cv::Size(30, 30),
      const cv::Size& maxSize = cv::Size()
    ) = 0;
};

/**
 * @brief Smart pointer type for face detector instances
 */
using FaceDetectorPtr = std::unique_ptr<IFaceDetector>;

#endif // IFACEDETECTOR_H
