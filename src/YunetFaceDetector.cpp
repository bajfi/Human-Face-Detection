// Copyright (c) 2025 JackLee
//
// This software is released under the MIT License.
// https://opensource.org/licenses/MIT

#include "YunetFaceDetector.h"
#include <opencv2/imgproc.hpp>
#include <filesystem>
#include <algorithm>
#ifdef OPENCV_DNN_CUDA
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudaarithm.hpp>
#endif

YunetFaceDetector::YunetFaceDetector()
{
    initializeCuda();
}

bool YunetFaceDetector::loadModel(const std::string& modelPath)
{
    if (modelPath.empty() || !isSupportedModelType(modelPath))
    {
        return false;
    }

    try
    {
        m_modelPath = modelPath;
        return recreateDetector();
    }
    catch (const cv::Exception& e)
    {
        m_modelPath.clear();
        return false;
    }
}

std::vector<DetectionResult> YunetFaceDetector::detectFaces(
  const cv::Mat& image
)
{
    std::vector<DetectionResult> results;

    if (image.empty() || !isLoaded())
    {
        return results;
    }

    try
    {
        // Validate image size - YuNet requires reasonable minimum dimensions
        // Very small images (< 64x64) can cause segfaults in CUDA backend
        const int MIN_SIZE = 64;
        if (image.rows < MIN_SIZE || image.cols < MIN_SIZE)
        {
            // Image too small for reliable detection, return empty results
            // gracefully
            return results;
        }

        // Ensure image is in correct format (BGR, 8-bit)
        cv::Mat processedImage;
        if (image.channels() == 1)
        {
            cv::cvtColor(image, processedImage, cv::COLOR_GRAY2BGR);
        }
        else if (image.channels() == 3)
        {
            processedImage = image;
        }
        else if (image.channels() == 4)
        {
            cv::cvtColor(image, processedImage, cv::COLOR_BGRA2BGR);
        }
        else
        {
            // Unsupported format
            return results;
        }

        // Ensure the image is 8-bit
        if (processedImage.depth() != CV_8U)
        {
            processedImage.convertTo(processedImage, CV_8U);
        }

        // Set input size to match current image
        m_detector->setInputSize(processedImage.size());

        // Perform detection
        cv::Mat faces;
        m_detector->detect(processedImage, faces);

        // Convert FaceDetectorYN output to our format
        results = convertOutput(faces, processedImage.size());
    }
    catch (const cv::Exception& e)
    {
        // Log the error for debugging if needed, but return empty results
        // gracefully
        results.clear();
    }
    catch (const std::exception& e)
    {
        // Handle any other exceptions gracefully
        results.clear();
    }

    return results;
}

bool YunetFaceDetector::isLoaded() const
{
    return !m_detector.empty();
}

std::string YunetFaceDetector::getModelInfo() const
{
    if (!isLoaded())
    {
        return "DNN Face Detector - not loaded";
    }

    std::filesystem::path fileInfo(m_modelPath);
    std::string info = "DNN Face Detector - " + fileInfo.filename().string();

    if (m_cudaAvailable)
    {
        info += " (CUDA ";
        info += (m_cudaEnabled ? "enabled" : "disabled");
        info += ")";
    }
    else
    {
        info += " (CPU only)";
    }

    return info;
}

std::vector<std::string> YunetFaceDetector::getSupportedExtensions() const
{
    return {".onnx", ".pb", ".caffemodel", ".t7", ".net", ".weights"};
}

void YunetFaceDetector::setDetectionParams(
  double scaleFactor,
  int minNeighbors,
  const cv::Size& minSize,
  const cv::Size& maxSize
)
{
    // For DNN, we only use minSize and maxSize
    // scaleFactor and minNeighbors are not applicable to DNN-based detection
    m_minSize =
      cv::Size(std::max(1, minSize.width), std::max(1, minSize.height));
    m_maxSize = maxSize;
}

void YunetFaceDetector::setConfidenceThreshold(float threshold)
{
    m_confidenceThreshold = std::clamp(threshold, 0.0f, 1.0f);
    if (m_detector && !m_detector.empty())
    {
        // Recreate detector with new threshold
        recreateDetector();
    }
}

void YunetFaceDetector::setNmsThreshold(float threshold)
{
    m_nmsThreshold = std::clamp(threshold, 0.0f, 1.0f);
    if (m_detector && !m_detector.empty())
    {
        // Recreate detector with new threshold
        recreateDetector();
    }
}

void YunetFaceDetector::setInputSize(const cv::Size& size)
{
    if (size.width > 0 && size.height > 0)
    {
        m_inputSize = size;
        if (m_detector && !m_detector.empty())
        {
            // Recreate detector with new input size
            recreateDetector();
        }
    }
}

std::vector<DetectionResult> YunetFaceDetector::convertOutput(
  const cv::Mat& faces, const cv::Size& imageSize
) const
{
    std::vector<DetectionResult> detections;

    if (faces.empty() || faces.rows == 0)
    {
        return detections;
    }

    // FaceDetectorYN output format: [N, 15] where N is number of detections
    // Each row: [x, y, w, h, landmarks..., confidence]
    // Confidence is at index 14
    for (int i = 0; i < faces.rows; ++i)
    {
        float x = faces.at<float>(i, 0);
        float y = faces.at<float>(i, 1);
        float w = faces.at<float>(i, 2);
        float h = faces.at<float>(i, 3);
        float confidence = faces.at<float>(i, 14);

        // Create bounding box
        cv::Rect boundingBox(
          static_cast<int>(x),
          static_cast<int>(y),
          static_cast<int>(w),
          static_cast<int>(h)
        );

        // Apply size constraints
        if (!m_minSize.empty() && (boundingBox.width < m_minSize.width ||
                                   boundingBox.height < m_minSize.height))
        {
            continue;
        }

        if (!m_maxSize.empty() && (boundingBox.width > m_maxSize.width ||
                                   boundingBox.height > m_maxSize.height))
        {
            continue;
        }

        // Ensure bounding box is within image bounds
        boundingBox &= cv::Rect(0, 0, imageSize.width, imageSize.height);

        if (boundingBox.width > 0 && boundingBox.height > 0)
        {
            detections.emplace_back(boundingBox, confidence);
        }
    }

    return detections;
}

bool YunetFaceDetector::isSupportedModelType(const std::string& modelPath) const
{
    std::filesystem::path fileInfo(modelPath);
    std::string extension = fileInfo.extension().string();

    const auto supportedExts = getSupportedExtensions();
    return std::any_of(
      supportedExts.begin(),
      supportedExts.end(),
      [&extension](const std::string& ext)
      {
          return extension == ext;
      }
    );
}

bool YunetFaceDetector::setCudaEnabled(bool enable)
{
    if (enable && !m_cudaAvailable)
    {
        return false; // CUDA not available
    }

    bool prevCudaEnabled = m_cudaEnabled;
    m_cudaEnabled = enable && m_cudaAvailable;

    // If CUDA state changed and we have a loaded model, recreate the detector
    if (prevCudaEnabled != m_cudaEnabled && !m_modelPath.empty())
    {
        return recreateDetector();
    }

    return true;
}

bool YunetFaceDetector::isCudaEnabled() const
{
    return m_cudaEnabled && m_cudaAvailable;
}

bool YunetFaceDetector::isCudaAvailable()
{
#ifdef OPENCV_DNN_CUDA
    try
    {
        return cv::cuda::getCudaEnabledDeviceCount() > 0;
    }
    catch (const cv::Exception&)
    {
        return false;
    }
#else
    return false;
#endif
}

void YunetFaceDetector::initializeCuda()
{
    m_cudaAvailable = isCudaAvailable();
    if (m_cudaAvailable)
    {
        m_cudaEnabled = true; // Enable CUDA by default if available
    }
}

bool YunetFaceDetector::recreateDetector()
{
    if (m_modelPath.empty())
    {
        return false;
    }

    // Helper lambda to create detector with given backend/target
    auto createDetector =
      [this](int backend, int target) -> cv::Ptr<cv::FaceDetectorYN>
    {
        try
        {
            return cv::FaceDetectorYN::create(
              m_modelPath,
              "",                    // config (empty for ONNX)
              m_inputSize,           // input size
              m_confidenceThreshold, // confidence threshold
              m_nmsThreshold,        // NMS threshold
              5000,                  // top K
              backend,
              target
            );
        }
        catch (...)
        {
            return cv::Ptr<cv::FaceDetectorYN>();
        }
    };

    cv::Ptr<cv::FaceDetectorYN> tempDetector;

    // Try CUDA first if enabled and available
    if (m_cudaEnabled && m_cudaAvailable)
    {
#ifdef OPENCV_DNN_CUDA
        tempDetector =
          createDetector(cv::dnn::DNN_BACKEND_CUDA, cv::dnn::DNN_TARGET_CUDA);

        if (!tempDetector || tempDetector.empty())
        {
            // CUDA creation failed, disable CUDA and fall back to CPU
            m_cudaEnabled = false;
        }
#else
        // CUDA not compiled in, disable it
        m_cudaEnabled = false;
#endif
    }

    // If CUDA failed or not enabled, try CPU backend
    if (!tempDetector || tempDetector.empty())
    {
        tempDetector =
          createDetector(cv::dnn::DNN_BACKEND_OPENCV, cv::dnn::DNN_TARGET_CPU);

        if (!tempDetector || tempDetector.empty())
        {
            // Both backends failed
            return false;
        }
    }

    // If we reach here, we have a valid detector
    m_detector = tempDetector;
    return true;
}
