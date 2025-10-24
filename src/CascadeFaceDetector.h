// Copyright (c) 2025 JackLee
//
// This software is released under the MIT License.
// https://opensource.org/licenses/MIT

#ifndef CASCADEFACEDETECTOR_H
#define CASCADEFACEDETECTOR_H

#include "IFaceDetector.h"
#include <opencv2/objdetect.hpp>
#ifdef OPENCV_DNN_CUDA
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudaarithm.hpp>
#include <opencv2/cudaobjdetect.hpp>
#endif

/**
 * @brief Face detector implementation using OpenCV CascadeClassifier
 *
 * This implementation uses traditional Haar Cascade classifiers for face
 * detection. While not as accurate as modern deep learning approaches, it
 * provides fast detection suitable for real-time applications with lower
 * computational requirements.
 *
 * Supported formats: XML files containing trained Haar cascade models
 */
class CascadeFaceDetector : public IFaceDetector
{
  public:
    CascadeFaceDetector();
    ~CascadeFaceDetector() override = default;

    // Disable copy constructor and assignment operator for resource management
    CascadeFaceDetector(const CascadeFaceDetector&) = delete;
    CascadeFaceDetector& operator=(const CascadeFaceDetector&) = delete;

    // Enable move semantics
    CascadeFaceDetector(CascadeFaceDetector&&) = default;
    CascadeFaceDetector& operator=(CascadeFaceDetector&&) = default;

    /**
     * @brief Load Haar Cascade model from XML file
     * @param modelPath Path to the XML cascade file
     * @return True if model loaded successfully, false otherwise
     */
    bool loadModel(const std::string& modelPath) override;

    /**
     * @brief Detect faces using Haar Cascade classifier
     * @param image Input image (BGR or grayscale)
     * @return Vector of detection results with bounding boxes
     */
    std::vector<DetectionResult> detectFaces(const cv::Mat& image) override;

    /**
     * @brief Check if cascade classifier is loaded
     * @return True if cascade is loaded and ready for detection
     */
    bool isLoaded() const override;

    /**
     * @brief Get information about the loaded cascade model
     * @return String describing the cascade status and file
     */
    std::string getModelInfo() const override;

    /**
     * @brief Get supported file extensions for cascade classifiers
     * @return Vector containing ".xml" extension
     */
    std::vector<std::string> getSupportedExtensions() const override;

    /**
     * @brief Configure cascade detection parameters
     * @param scaleFactor How much the image size is reduced at each scale
     * @param minNeighbors How many neighbors each candidate rectangle should
     * retain
     * @param minSize Minimum possible object size. Smaller objects are ignored
     * @param maxSize Maximum possible object size
     */
    void setDetectionParams(
      double scaleFactor = 1.1,
      int minNeighbors = 3,
      const cv::Size& minSize = cv::Size(30, 30),
      const cv::Size& maxSize = cv::Size()
    ) override;

    /**
     * @brief Enable or disable CUDA acceleration if available
     * @param enable True to enable CUDA, false to use CPU
     * @return True if CUDA was successfully enabled/disabled
     */
    bool setCudaEnabled(bool enable);

    /**
     * @brief Check if CUDA acceleration is currently enabled
     * @return True if CUDA is enabled and available
     */
    bool isCudaEnabled() const;

    /**
     * @brief Check if CUDA is available for cascade detection
     * @return True if CUDA devices and OpenCV GPU support are available
     */
    static bool isCudaAvailable();

  private:
    cv::CascadeClassifier m_cascade;
    std::string m_modelPath;

    // Detection parameters
    double m_scaleFactor = 1.1;
    int m_minNeighbors = 3;
    cv::Size m_minSize = cv::Size(30, 30);
    cv::Size m_maxSize = cv::Size();

    // GPU-specific members
    bool m_cudaEnabled = false;
    bool m_cudaAvailable = false;

#ifdef OPENCV_DNN_CUDA
    mutable cv::cuda::GpuMat m_gpuImagePool; // GPU memory pool for input image
    mutable cv::cuda::GpuMat m_gpuGrayPool;  // GPU memory pool for grayscale
    mutable cv::cuda::Stream m_stream;       // CUDA stream for async operations
    // Note: Using GPU for preprocessing only, CPU for cascade detection
    // Full GPU cascade detection is deprecated in newer OpenCV versions
#endif

    /**
     * @brief Preprocess image for cascade detection
     * @param image Input image
     * @return Preprocessed grayscale image
     */
    cv::Mat preprocessImage(const cv::Mat& image) const;

    /**
     * @brief Initialize CUDA capability detection
     */
    void initializeCuda();

    /**
     * @brief Recreate cascade classifier with current CUDA settings
     * @return True if classifier was successfully recreated
     */
    bool recreateClassifier();

    /**
     * @brief GPU-accelerated face detection
     * @param image Input image
     * @return Vector of detection results
     */
    std::vector<DetectionResult> detectFacesGpu(const cv::Mat& image);

    /**
     * @brief CPU fallback detection (existing implementation)
     * @param image Input image
     * @return Vector of detection results
     */
    std::vector<DetectionResult> detectFacesCpu(const cv::Mat& image);

    /**
     * @brief Ensure GPU memory pools are properly sized
     * @param imageSize Size of the input image
     */
    void ensureGpuMemory(const cv::Size& imageSize) const;
};

#endif // CASCADEFACEDETECTOR_H
