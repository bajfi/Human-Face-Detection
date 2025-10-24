// Copyright (c) 2025 JackLee
//
// This software is released under the MIT License.
// https://opensource.org/licenses/MIT

#include "CascadeFaceDetector.h"
#include <opencv2/imgproc.hpp>
#include <filesystem>

CascadeFaceDetector::CascadeFaceDetector()
{
    initializeCuda();
}

bool CascadeFaceDetector::loadModel(const std::string& modelPath)
{
    if (modelPath.empty())
    {
        return false;
    }

    m_modelPath = modelPath;
    return recreateClassifier();
}

std::vector<DetectionResult> CascadeFaceDetector::detectFaces(
  const cv::Mat& image
)
{
    if (image.empty() || !isLoaded())
    {
        return {};
    }

    // Use GPU detection if enabled and available
    if (m_cudaEnabled && m_cudaAvailable)
    {
        return detectFacesGpu(image);
    }

    // Fall back to CPU detection
    return detectFacesCpu(image);
}

bool CascadeFaceDetector::isLoaded() const
{
    return !m_cascade.empty();
}

std::string CascadeFaceDetector::getModelInfo() const
{
    if (!isLoaded())
    {
        return "Haar Cascade Classifier - not loaded";
    }

    std::filesystem::path fileInfo(m_modelPath);
    std::string info =
      "Haar Cascade Classifier - " + fileInfo.filename().string();

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

std::vector<std::string> CascadeFaceDetector::getSupportedExtensions() const
{
    return {".xml", ".cascade"};
}

void CascadeFaceDetector::setDetectionParams(
  double scaleFactor,
  int minNeighbors,
  const cv::Size& minSize,
  const cv::Size& maxSize
)
{
    // Validate parameters
    m_scaleFactor = std::max(1.01, scaleFactor);
    m_minNeighbors = std::max(1, minNeighbors);
    m_minSize =
      cv::Size(std::max(1, minSize.width), std::max(1, minSize.height));
    m_maxSize = maxSize;
}

cv::Mat CascadeFaceDetector::preprocessImage(const cv::Mat& image) const
{
    cv::Mat processedImage;

    // Convert to grayscale if necessary
    if (image.channels() > 1)
    {
        cv::cvtColor(image, processedImage, cv::COLOR_BGR2GRAY);
    }
    else
    {
        processedImage = image.clone();
    }

    // Equalize histogram for better detection performance
    // This helps normalize lighting conditions
    cv::equalizeHist(processedImage, processedImage);

    return processedImage;
}

void CascadeFaceDetector::initializeCuda()
{
#ifdef OPENCV_DNN_CUDA
    try
    {
        int deviceCount = cv::cuda::getCudaEnabledDeviceCount();
        m_cudaAvailable = (deviceCount > 0);

        if (m_cudaAvailable)
        {
            // Test basic GPU functionality
            cv::cuda::GpuMat testMat(100, 100, CV_8UC1);
            testMat.setTo(cv::Scalar(0));
            // If we reach here, CUDA is functional
        }
    }
    catch (const cv::Exception& e)
    {
        m_cudaAvailable = false;
    }
#else
    m_cudaAvailable = false;
#endif
    m_cudaEnabled = false; // Start with CPU by default
}

bool CascadeFaceDetector::recreateClassifier()
{
    // Load CPU cascade
    cv::CascadeClassifier tempCascade;
    if (!tempCascade.load(m_modelPath))
    {
        m_modelPath.clear();
        return false;
    }
    m_cascade = std::move(tempCascade);

    // GPU acceleration is available for preprocessing only
    // Actual cascade detection runs on CPU with GPU-preprocessed image
    return true;
}

std::vector<DetectionResult> CascadeFaceDetector::detectFacesGpu(
  const cv::Mat& image
)
{
#ifdef OPENCV_DNN_CUDA
    std::vector<DetectionResult> results;

    try
    {
        // Ensure GPU memory pools are properly sized
        ensureGpuMemory(image.size());

        // Upload image to GPU for accelerated preprocessing
        m_gpuImagePool.upload(image, m_stream);

        // Convert to grayscale on GPU if necessary
        cv::cuda::GpuMat* grayPtr = &m_gpuGrayPool;
        if (image.channels() > 1)
        {
            cv::cuda::cvtColor(
              m_gpuImagePool, m_gpuGrayPool, cv::COLOR_BGR2GRAY, 0, m_stream
            );
        }
        else
        {
            m_gpuGrayPool = m_gpuImagePool;
            grayPtr = &m_gpuGrayPool;
        }

        // Equalize histogram on GPU for better detection performance
        cv::cuda::equalizeHist(*grayPtr, m_gpuGrayPool, m_stream);

        // Wait for GPU operations to complete
        m_stream.waitForCompletion();

        // Download preprocessed image back to CPU for cascade detection
        cv::Mat processedImage;
        m_gpuGrayPool.download(processedImage);

        // Perform cascade detection on CPU with GPU-preprocessed image
        std::vector<cv::Rect> faces;
        m_cascade.detectMultiScale(
          processedImage,
          faces,
          m_scaleFactor,
          m_minNeighbors,
          0, // flags (deprecated, should be 0)
          m_minSize,
          m_maxSize.empty() ? cv::Size() : m_maxSize
        );

        // Convert results
        results.reserve(faces.size());
        for (const auto& face : faces)
        {
            results.emplace_back(
              face, 1.0f
            ); // Cascade doesn't provide confidence
        }
    }
    catch (const cv::Exception& e)
    {
        // GPU preprocessing failed, fall back to CPU
        results = detectFacesCpu(image);
        // Optionally disable CUDA on repeated failures
    }

    return results;
#else
    // Fallback to CPU if CUDA not available at compile time
    return detectFacesCpu(image);
#endif
}

std::vector<DetectionResult> CascadeFaceDetector::detectFacesCpu(
  const cv::Mat& image
)
{
    std::vector<DetectionResult> results;

    // Preprocess image (convert to grayscale and equalize histogram)
    cv::Mat processedImage = preprocessImage(image);

    // Detect faces using cascade classifier
    std::vector<cv::Rect> faces;
    m_cascade.detectMultiScale(
      processedImage,
      faces,
      m_scaleFactor,
      m_minNeighbors,
      0, // flags (deprecated, should be 0)
      m_minSize,
      m_maxSize.empty() ? cv::Size() : m_maxSize
    );

    // Convert cv::Rect to DetectionResult
    results.reserve(faces.size());
    for (const auto& face : faces)
    {
        // Cascade classifiers don't provide confidence scores,
        // so we use a default confidence of 1.0
        results.emplace_back(face, 1.0f);
    }

    return results;
}

void CascadeFaceDetector::ensureGpuMemory(const cv::Size& imageSize) const
{
#ifdef OPENCV_DNN_CUDA
    // Resize GPU memory pools if necessary
    if (m_gpuImagePool.size() != imageSize || m_gpuImagePool.empty())
    {
        m_gpuImagePool.create(imageSize, CV_8UC3);
        m_gpuGrayPool.create(imageSize, CV_8UC1);
    }
#endif
}

bool CascadeFaceDetector::setCudaEnabled(bool enable)
{
    if (!m_cudaAvailable)
    {
        return false; // CUDA not available
    }

    if (m_cudaEnabled == enable)
    {
        return true; // Already in desired state
    }

    m_cudaEnabled = enable;

    // Recreate classifier to apply CUDA settings
    if (!m_modelPath.empty())
    {
        return recreateClassifier();
    }

    return true;
}

bool CascadeFaceDetector::isCudaEnabled() const
{
    return m_cudaEnabled && m_cudaAvailable;
}

bool CascadeFaceDetector::isCudaAvailable()
{
#ifdef OPENCV_DNN_CUDA
    try
    {
        static bool checked = false;
        static bool available = false;

        if (!checked)
        {
            int deviceCount = cv::cuda::getCudaEnabledDeviceCount();
            available = (deviceCount > 0);

            if (available)
            {
                // Test GPU cascade loading capability
                try
                {
                    cv::cuda::GpuMat testMat(10, 10, CV_8UC1);
                    testMat.setTo(cv::Scalar(0));
                }
                catch (...)
                {
                    available = false;
                }
            }
            checked = true;
        }

        return available;
    }
    catch (...)
    {
        return false;
    }
#else
    return false;
#endif
}
