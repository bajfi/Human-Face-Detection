// Copyright (c) 2025 JackLee
//
// This software is released under the MIT License.
// https://opensource.org/licenses/MIT

#include "FaceDetectorFactory.h"
#include <filesystem>
#include <string>
#include <algorithm>
#include <opencv2/dnn.hpp>

FaceDetectorPtr FaceDetectorFactory::createDetector(
  const std::string& modelPath
)
{
    DetectorType type = detectModelType(modelPath);
    return createDetector(type);
}

FaceDetectorPtr FaceDetectorFactory::createDetector(DetectorType type)
{
    switch (type)
    {
    case DetectorType::CASCADE:
        return std::make_unique<CascadeFaceDetector>();

    case DetectorType::DNN:
        if (isDnnAvailable())
        {
            return std::make_unique<YunetFaceDetector>();
        }
        return nullptr;

    case DetectorType::AUTO_DETECT:
        // For auto-detect without a specific file, default to cascade
        return std::make_unique<CascadeFaceDetector>();

    case DetectorType::UNKNOWN:
    default:
        return nullptr;
    }
}

FaceDetectorFactory::DetectorType FaceDetectorFactory::detectModelType(
  const std::string& modelPath
)
{
    if (modelPath.empty())
    {
        return DetectorType::UNKNOWN;
    }

    std::string extension = getFileExtension(modelPath);

    // Check for Haar Cascade files
    if (extension == ".xml" || extension == ".cascade")
    {
        return DetectorType::CASCADE;
    }

    // Check for DNN model files
    const std::vector<std::string> dnnExtensions = {
      ".onnx", ".pb", ".caffemodel", ".t7", ".net", ".weights"
    };
    if (std::find(dnnExtensions.begin(), dnnExtensions.end(), extension) !=
        dnnExtensions.end())
    {
        return isDnnAvailable() ? DetectorType::DNN : DetectorType::UNKNOWN;
    }

    return DetectorType::UNKNOWN;
}

bool FaceDetectorFactory::isModelSupported(const std::string& modelPath)
{
    return detectModelType(modelPath) != DetectorType::UNKNOWN;
}

std::vector<FaceDetectorFactory::ModelInfo> FaceDetectorFactory::
  getAvailableDetectors()
{
    std::vector<ModelInfo> detectors;

    // Haar Cascade Classifier - always available
    detectors.emplace_back(
      DetectorType::CASCADE,
      "Haar Cascade Classifier - Fast, traditional method suitable for "
      "real-time applications",
      std::vector<std::string>{".xml", ".cascade"},
      true
    );

    // DNN Face Detector - check if DNN module is available
    detectors.emplace_back(
      DetectorType::DNN,
      "Deep Neural Network - High accuracy, modern approach (YuNet, etc.)",
      std::vector<std::string>{
        ".onnx", ".pb", ".caffemodel", ".t7", ".net", ".weights"
      },
      isDnnAvailable()
    );

    return detectors;
}

std::vector<std::string> FaceDetectorFactory::getAllSupportedExtensions()
{
    std::vector<std::string> allExtensions;

    auto detectors = getAvailableDetectors();
    for (const auto& detector : detectors)
    {
        if (detector.isAvailable)
        {
            allExtensions.insert(
              allExtensions.end(),
              detector.supportedExtensions.begin(),
              detector.supportedExtensions.end()
            );
        }
    }

    // Remove duplicates and sort
    std::sort(allExtensions.begin(), allExtensions.end());
    allExtensions.erase(
      std::unique(allExtensions.begin(), allExtensions.end()),
      allExtensions.end()
    );

    return allExtensions;
}

FaceDetectorPtr FaceDetectorFactory::createOptimizedDetector(
  const std::string& modelPath
)
{
    auto detector = createDetector(modelPath);

    if (!detector)
    {
        return nullptr;
    }

    DetectorType type = detectModelType(modelPath);
    configureDetector(detector.get(), type);

    // Attempt to load the model
    if (!detector->loadModel(modelPath))
    {
        return nullptr;
    }

    return detector;
}

FaceDetectorFactory::DetectorType FaceDetectorFactory::getRecommendedDetector(
  bool prioritizeSpeed
)
{
    if (prioritizeSpeed)
    {
        // Cascade is faster but less accurate
        return DetectorType::CASCADE;
    }
    else
    {
        // DNN is more accurate but slower
        return isDnnAvailable() ? DetectorType::DNN : DetectorType::CASCADE;
    }
}

std::string FaceDetectorFactory::getFileExtension(const std::string& filePath)
{
    std::filesystem::path fileInfo(filePath);
    std::string extension = fileInfo.extension().string();

    return extension;
}

bool FaceDetectorFactory::isDnnAvailable()
{
    try
    {
        // Try to create a simple DNN network to check if the module is
        // available
        cv::dnn::Net net;
        return true;
    }
    catch (...)
    {
        return false;
    }
}

void FaceDetectorFactory::configureDetector(
  IFaceDetector* detector, DetectorType type
)
{
    if (!detector)
    {
        return;
    }

    switch (type)
    {
    case DetectorType::CASCADE: {
        // Optimize cascade parameters for balance between speed and accuracy
        detector->setDetectionParams(
          1.1,               // scaleFactor
          3,                 // minNeighbors
          cv::Size(30, 30),  // minSize
          cv::Size(300, 300) // maxSize
        );
        break;
    }

    case DetectorType::DNN: {
        // Configure DNN detector
        detector->setDetectionParams(
          1.0,               // scaleFactor (not used for DNN)
          1,                 // minNeighbors (not used for DNN)
          cv::Size(20, 20),  // minSize - smaller for better detection
          cv::Size(500, 500) // maxSize
        );

        // Set DNN-specific parameters if it's a YunetFaceDetector
        if (auto dnnDetector = dynamic_cast<YunetFaceDetector*>(detector))
        {
            dnnDetector->setConfidenceThreshold(0.6f);
            dnnDetector->setNmsThreshold(0.3f);
            dnnDetector->setInputSize(cv::Size(320, 320)); // YuNet default
        }
        break;
    }

    default:
        // Use default parameters
        break;
    }
}
