// Copyright (c) 2025 JackLee
//
// This software is released under the MIT License.
// https://opensource.org/licenses/MIT

#ifndef TEST_UTILS_H
#define TEST_UTILS_H

#include <opencv2/opencv.hpp>
#include <vector>
#include <string>
#include <filesystem>
#include "IFaceDetector.h"

namespace TestUtils
{

/**
 * @brief Test configuration constants
 */
struct TestConfig
{
    static constexpr size_t EXPECTED_FACE_COUNT = 3;
    static constexpr size_t FACE_COUNT_TOLERANCE = 1;
    static constexpr float DEFAULT_CONFIDENCE_THRESHOLD = 0.6f;
    static constexpr float DEFAULT_NMS_THRESHOLD = 0.3f;
    static constexpr float CONFIDENCE_EPSILON = 0.01f;
    static constexpr int BBOX_COORDINATE_TOLERANCE = 2;
};

/**
 * @brief Test file paths
 */
struct TestPaths
{
    static inline const std::string CASCADE_MODEL =
      "models/haarcascade_frontalface_alt.xml";
    static inline const std::string DNN_MODEL =
      "models/face_detection_yunet_2023mar.onnx";
    static inline const std::string TEST_IMAGE = "test_images/test01.jpg";
    static inline const std::string INVALID_MODEL = "models/nonexistent.xml";
};

/**
 * @brief Validation utilities for test results
 */
class ValidationUtils
{
  public:
    /**
     * @brief Validate a single detection result
     */
    static bool isValidDetection(
      const DetectionResult& result, const cv::Size& imageSize
    )
    {
        const auto& box = result.boundingBox;
        return box.x >= 0 && box.y >= 0 &&
               box.x + box.width <= imageSize.width &&
               box.y + box.height <= imageSize.height && box.width > 0 &&
               box.height > 0 && result.confidence >= 0.0f &&
               result.confidence <= 1.0f;
    }

    /**
     * @brief Check if detection count is within expected range
     */
    static bool isExpectedFaceCount(
      size_t detectedCount,
      size_t expectedCount,
      size_t tolerance = TestConfig::FACE_COUNT_TOLERANCE
    )
    {
        return detectedCount >= (expectedCount - tolerance) &&
               detectedCount <= (expectedCount + tolerance);
    }

    /**
     * @brief Validate all detections in a result set
     */
    static bool areAllDetectionsValid(
      const std::vector<DetectionResult>& results, const cv::Size& imageSize
    )
    {
        for (const auto& result : results)
        {
            if (!isValidDetection(result, imageSize))
            {
                return false;
            }
        }
        return true;
    }

    /**
     * @brief Check if confidence scores are reasonable for DNN detector
     */
    static bool hasReasonableConfidenceScores(
      const std::vector<DetectionResult>& results, float minConfidence = 0.1f
    )
    {
        for (const auto& result : results)
        {
            if (result.confidence < minConfidence || result.confidence > 1.0f)
            {
                return false;
            }
        }
        return true;
    }

    /**
     * @brief Calculate Intersection over Union (IoU) for two bounding boxes
     */
    static float calculateIoU(const cv::Rect& box1, const cv::Rect& box2)
    {
        cv::Rect intersection = box1 & box2;
        cv::Rect unionRect = box1 | box2;

        if (unionRect.area() == 0)
        {
            return 0.0f;
        }

        return static_cast<float>(intersection.area()) / unionRect.area();
    }

    /**
     * @brief Find overlapping detections (for NMS testing)
     */
    static bool hasOverlappingDetections(
      const std::vector<DetectionResult>& results, float threshold = 0.5f
    )
    {
        for (size_t i = 0; i < results.size(); ++i)
        {
            for (size_t j = i + 1; j < results.size(); ++j)
            {
                float iou =
                  calculateIoU(results[i].boundingBox, results[j].boundingBox);
                if (iou > threshold)
                {
                    return true;
                }
            }
        }
        return false;
    }

    /**
     * @brief Compare two detection results for similarity (within tolerance)
     */
    static bool areDetectionsSimilar(
      const DetectionResult& det1,
      const DetectionResult& det2,
      int coordinateTolerance = TestConfig::BBOX_COORDINATE_TOLERANCE,
      float confidenceTolerance = TestConfig::CONFIDENCE_EPSILON
    )
    {
        const cv::Rect& box1 = det1.boundingBox;
        const cv::Rect& box2 = det2.boundingBox;

        return std::abs(box1.x - box2.x) <= coordinateTolerance &&
               std::abs(box1.y - box2.y) <= coordinateTolerance &&
               std::abs(box1.width - box2.width) <= coordinateTolerance &&
               std::abs(box1.height - box2.height) <= coordinateTolerance &&
               std::abs(det1.confidence - det2.confidence) <=
                 confidenceTolerance;
    }
};

/**
 * @brief Image processing utilities for testing
 */
class ImageUtils
{
  public:
    /**
     * @brief Load test image and validate it exists
     */
    static cv::Mat loadTestImage(
      const std::string& imagePath = TestPaths::TEST_IMAGE
    )
    {
        if (!std::filesystem::exists(imagePath))
        {
            throw std::runtime_error("Test image not found: " + imagePath);
        }

        cv::Mat image = cv::imread(imagePath);
        if (image.empty())
        {
            throw std::runtime_error("Failed to load test image: " + imagePath);
        }

        return image;
    }

    /**
     * @brief Create synthetic test images for edge case testing
     */
    static cv::Mat createSyntheticImage(
      const cv::Size& size,
      int type = CV_8UC3,
      const cv::Scalar& color = cv::Scalar(128, 128, 128)
    )
    {
        return cv::Mat(size, type, color);
    }

    /**
     * @brief Convert image to grayscale and back to BGR for testing
     */
    static cv::Mat convertToGrayscaleBGR(const cv::Mat& image)
    {
        cv::Mat gray, grayBGR;
        cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);
        cv::cvtColor(gray, grayBGR, cv::COLOR_GRAY2BGR);
        return grayBGR;
    }

    /**
     * @brief Resize image while maintaining aspect ratio
     */
    static cv::Mat resizeWithAspectRatio(
      const cv::Mat& image, const cv::Size& targetSize
    )
    {
        cv::Mat resized;
        cv::resize(image, resized, targetSize, 0, 0, cv::INTER_LINEAR);
        return resized;
    }
};

/**
 * @brief File system utilities for testing
 */
class FileUtils
{
  public:
    /**
     * @brief Check if all required test files exist
     */
    static bool validateTestEnvironment()
    {
        return std::filesystem::exists(TestPaths::CASCADE_MODEL) &&
               std::filesystem::exists(TestPaths::DNN_MODEL) &&
               std::filesystem::exists(TestPaths::TEST_IMAGE);
    }

    /**
     * @brief Get list of missing test files
     */
    static std::vector<std::string> getMissingTestFiles()
    {
        std::vector<std::string> missing;

        if (!std::filesystem::exists(TestPaths::CASCADE_MODEL))
        {
            missing.push_back(TestPaths::CASCADE_MODEL);
        }
        if (!std::filesystem::exists(TestPaths::DNN_MODEL))
        {
            missing.push_back(TestPaths::DNN_MODEL);
        }
        if (!std::filesystem::exists(TestPaths::TEST_IMAGE))
        {
            missing.push_back(TestPaths::TEST_IMAGE);
        }

        return missing;
    }

    /**
     * @brief Validate model file extension
     */
    static bool hasValidModelExtension(
      const std::string& modelPath,
      const std::vector<std::string>& validExtensions
    )
    {
        std::filesystem::path path(modelPath);
        std::string extension = path.extension().string();

        return std::find(
                 validExtensions.begin(), validExtensions.end(), extension
               ) != validExtensions.end();
    }
};

/**
 * @brief Performance measurement utilities
 */
class PerformanceUtils
{
  public:
    /**
     * @brief Measure detection time for performance testing
     */
    template <typename DetectorType>
    static double measureDetectionTime(
      DetectorType& detector, const cv::Mat& image, int iterations = 10
    )
    {
        auto start = std::chrono::high_resolution_clock::now();

        for (int i = 0; i < iterations; ++i)
        {
            auto results = detector.detectFaces(image);
            cv::getTickCount(); // Prevent compiler optimization
        }

        auto end = std::chrono::high_resolution_clock::now();
        auto duration =
          std::chrono::duration_cast<std::chrono::microseconds>(end - start);

        return static_cast<double>(duration.count()) /
               (iterations * 1000.0); // Return ms
    }

    /**
     * @brief Measure model loading time
     */
    template <typename DetectorType>
    static double measureLoadingTime(
      DetectorType& detector, const std::string& modelPath
    )
    {
        auto start = std::chrono::high_resolution_clock::now();
        bool result = detector.loadModel(modelPath);
        auto end = std::chrono::high_resolution_clock::now();

        if (!result)
        {
            return -1.0; // Indicate failure
        }

        auto duration =
          std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        return static_cast<double>(duration.count()) / 1000.0; // Return ms
    }
};

} // namespace TestUtils

#endif // TEST_UTILS_H
