// Copyright (c) 2025 JackLee
//
// This software is released under the MIT License.
// https://opensource.org/licenses/MIT

#include <gtest/gtest.h>
#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <filesystem>
#include <memory>
#include <algorithm>
#include "YunetFaceDetector.h"

/**
 * @brief Test fixture for YunetFaceDetector testing
 *
 * This test fixture provides common setup and utilities for all YuNet detector
 * tests. It handles test data paths, image loading, YuNet model configuration,
 * and validation helpers.
 */
class YunetFaceDetectorTest : public ::testing::Test
{
  protected:
    void SetUp() override
    {
        // Setup test data paths
        modelPath = "models/face_detection_yunet_2023mar.onnx";
        testImagePath = "test_images/test01.jpg";
        invalidModelPath = "models/nonexistent.onnx";

        // Create detector instance
        detector = std::make_unique<YunetFaceDetector>();

        // Verify test files exist
        ASSERT_TRUE(std::filesystem::exists(testImagePath))
          << "Test image not found: " << testImagePath;
        ASSERT_TRUE(std::filesystem::exists(modelPath))
          << "Model file not found: " << modelPath;

        // Load test image
        testImage = cv::imread(testImagePath);
        ASSERT_FALSE(testImage.empty())
          << "Failed to load test image: " << testImagePath;
    }

    void TearDown() override
    {
        detector.reset();
    }

    /**
     * @brief Helper function to validate detection results
     */
    bool isValidDetection(
      const DetectionResult& result, const cv::Size& imageSize
    ) const
    {
        const auto& box = result.boundingBox;
        return box.x >= 0 && box.y >= 0 &&
               box.x + box.width <= imageSize.width &&
               box.y + box.height <= imageSize.height && box.width > 0 &&
               box.height > 0 && result.confidence >= 0.0f &&
               result.confidence <= 1.0f;
    }

    /**
     * @brief Helper to count faces within expected range (allowing for
     * detection variance)
     */
    bool isExpectedFaceCount(
      size_t detectedCount, size_t expectedCount, size_t tolerance = 1
    ) const
    {
        return detectedCount >= (expectedCount - tolerance) &&
               detectedCount <= (expectedCount + tolerance);
    }

    /**
     * @brief Helper to check if confidence scores are reasonable for DNN
     * detector
     */
    bool hasReasonableConfidenceScores(
      const std::vector<DetectionResult>& results
    ) const
    {
        for (const auto& result : results)
        {
            if (result.confidence < 0.1f || result.confidence > 1.0f)
            {
                return false;
            }
        }
        return true;
    }

    /**
     * @brief Helper to find overlapping detections (for NMS testing)
     */
    bool hasOverlappingDetections(
      const std::vector<DetectionResult>& results, float threshold = 0.5f
    ) const
    {
        for (size_t i = 0; i < results.size(); ++i)
        {
            for (size_t j = i + 1; j < results.size(); ++j)
            {
                cv::Rect intersection =
                  results[i].boundingBox & results[j].boundingBox;
                cv::Rect unionRect =
                  results[i].boundingBox | results[j].boundingBox;

                float iou =
                  static_cast<float>(intersection.area()) / unionRect.area();
                if (iou > threshold)
                {
                    return true;
                }
            }
        }
        return false;
    }

  protected:
    std::unique_ptr<YunetFaceDetector> detector;
    std::string modelPath;
    std::string testImagePath;
    std::string invalidModelPath;
    cv::Mat testImage;

    // Expected results for test01.jpg with YuNet
    static constexpr size_t EXPECTED_FACE_COUNT = 3;
    static constexpr size_t FACE_COUNT_TOLERANCE = 1;
    static constexpr float DEFAULT_CONFIDENCE_THRESHOLD = 0.6f;
    static constexpr float DEFAULT_NMS_THRESHOLD = 0.3f;
};

// ============================================================================
// Model Loading Tests
// ============================================================================

TEST_F(YunetFaceDetectorTest, LoadValidModel_Success)
{
    bool result = detector->loadModel(modelPath);

    EXPECT_TRUE(result);
    EXPECT_TRUE(detector->isLoaded());
    EXPECT_FALSE(detector->getModelInfo().empty());
    std::string modelInfo = detector->getModelInfo();
    EXPECT_TRUE(
      modelInfo.find("yunet") != std::string::npos ||
      modelInfo.find("loaded") != std::string::npos ||
      modelInfo.find("DNN") != std::string::npos
    );
}

TEST_F(YunetFaceDetectorTest, LoadInvalidModel_Failure)
{
    bool result = detector->loadModel(invalidModelPath);

    EXPECT_FALSE(result);
    EXPECT_FALSE(detector->isLoaded());
}

TEST_F(YunetFaceDetectorTest, LoadEmptyPath_Failure)
{
    bool result = detector->loadModel("");

    EXPECT_FALSE(result);
    EXPECT_FALSE(detector->isLoaded());
}

TEST_F(YunetFaceDetectorTest, LoadUnsupportedFormat_Failure)
{
    bool result = detector->loadModel("test.txt");

    EXPECT_FALSE(result);
    EXPECT_FALSE(detector->isLoaded());
}

TEST_F(YunetFaceDetectorTest, LoadModelTwice_SecondLoadSucceeds)
{
    // Load valid model first
    ASSERT_TRUE(detector->loadModel(modelPath));
    ASSERT_TRUE(detector->isLoaded());

    // Load same model again
    bool result = detector->loadModel(modelPath);

    EXPECT_TRUE(result);
    EXPECT_TRUE(detector->isLoaded());
}

// ============================================================================
// Face Detection Tests
// ============================================================================

TEST_F(YunetFaceDetectorTest, DetectFaces_ValidImage_DetectsExpectedFaces)
{
    ASSERT_TRUE(detector->loadModel(modelPath));

    auto results = detector->detectFaces(testImage);

    EXPECT_TRUE(isExpectedFaceCount(
      results.size(), EXPECTED_FACE_COUNT, FACE_COUNT_TOLERANCE
    ));
    EXPECT_TRUE(hasReasonableConfidenceScores(results));

    // Validate all detections
    for (const auto& result : results)
    {
        EXPECT_TRUE(isValidDetection(result, testImage.size()));
        EXPECT_GE(result.confidence, DEFAULT_CONFIDENCE_THRESHOLD);
    }
}

TEST_F(YunetFaceDetectorTest, DetectFaces_EmptyImage_ReturnsEmpty)
{
    ASSERT_TRUE(detector->loadModel(modelPath));

    cv::Mat emptyImage;
    auto results = detector->detectFaces(emptyImage);

    EXPECT_TRUE(results.empty());
}

TEST_F(YunetFaceDetectorTest, DetectFaces_NoModelLoaded_ReturnsEmpty)
{
    // Don't load model
    ASSERT_FALSE(detector->isLoaded());

    auto results = detector->detectFaces(testImage);

    EXPECT_TRUE(results.empty());
}

TEST_F(YunetFaceDetectorTest, DetectFaces_GrayscaleImage_WorksCorrectly)
{
    ASSERT_TRUE(detector->loadModel(modelPath));

    cv::Mat grayImage;
    cv::cvtColor(testImage, grayImage, cv::COLOR_BGR2GRAY);

    // Convert back to 3-channel for DNN (YuNet expects BGR)
    cv::Mat grayBGR;
    cv::cvtColor(grayImage, grayBGR, cv::COLOR_GRAY2BGR);

    auto results = detector->detectFaces(grayBGR);

    EXPECT_TRUE(isExpectedFaceCount(
      results.size(), EXPECTED_FACE_COUNT, FACE_COUNT_TOLERANCE
    ));
}

TEST_F(YunetFaceDetectorTest, DetectFaces_DifferentImageSizes_HandledCorrectly)
{
    ASSERT_TRUE(detector->loadModel(modelPath));

    // Test with resized image
    cv::Mat resizedImage;
    cv::resize(testImage, resizedImage, cv::Size(640, 480));

    auto results = detector->detectFaces(resizedImage);

    EXPECT_FALSE(results.empty());
    for (const auto& result : results)
    {
        EXPECT_TRUE(isValidDetection(result, resizedImage.size()));
    }
}

// ============================================================================
// Confidence Threshold Tests
// ============================================================================

TEST_F(YunetFaceDetectorTest, SetConfidenceThreshold_AffectsDetectionCount)
{
    ASSERT_TRUE(detector->loadModel(modelPath));

    // Test with low threshold (should detect more faces)
    detector->setConfidenceThreshold(0.3f);
    auto lowThresholdResults = detector->detectFaces(testImage);

    // Test with high threshold (should detect fewer faces)
    detector->setConfidenceThreshold(0.9f);
    auto highThresholdResults = detector->detectFaces(testImage);

    // Low threshold should detect more or equal faces
    EXPECT_GE(lowThresholdResults.size(), highThresholdResults.size());

    // All high threshold detections should have high confidence
    for (const auto& result : highThresholdResults)
    {
        EXPECT_GE(result.confidence, 0.9f);
    }
}

TEST_F(YunetFaceDetectorTest, SetConfidenceThreshold_ExtremeLow_DetectsMany)
{
    ASSERT_TRUE(detector->loadModel(modelPath));

    detector->setConfidenceThreshold(0.01f);
    auto results = detector->detectFaces(testImage);

    // Should detect at least the expected faces (possibly more false positives)
    EXPECT_GE(results.size(), EXPECTED_FACE_COUNT);
}

TEST_F(YunetFaceDetectorTest, SetConfidenceThreshold_ExtremeHigh_DetectsFew)
{
    ASSERT_TRUE(detector->loadModel(modelPath));

    detector->setConfidenceThreshold(0.99f);
    auto results = detector->detectFaces(testImage);

    // Might detect very few or no faces with such high threshold
    for (const auto& result : results)
    {
        EXPECT_GE(result.confidence, 0.99f);
    }
}

// ============================================================================
// NMS Threshold Tests
// ============================================================================

TEST_F(YunetFaceDetectorTest, SetNmsThreshold_AffectsOverlappingDetections)
{
    ASSERT_TRUE(detector->loadModel(modelPath));

    // Low NMS threshold (aggressive suppression - fewer overlaps)
    detector->setNmsThreshold(0.1f);
    detector->setConfidenceThreshold(
      0.3f
    ); // Low confidence to get more detections
    auto lowNmsResults = detector->detectFaces(testImage);

    // High NMS threshold (less suppression - more overlaps allowed)
    detector->setNmsThreshold(0.8f);
    auto highNmsResults = detector->detectFaces(testImage);

    // High NMS threshold might allow more detections (including overlapping
    // ones)
    EXPECT_GE(highNmsResults.size(), lowNmsResults.size());
}

TEST_F(YunetFaceDetectorTest, SetNmsThreshold_Zero_MaximalSuppression)
{
    ASSERT_TRUE(detector->loadModel(modelPath));

    detector->setNmsThreshold(0.0f);
    detector->setConfidenceThreshold(0.3f);
    auto results = detector->detectFaces(testImage);

    // Should have minimal overlapping detections
    EXPECT_FALSE(hasOverlappingDetections(results, 0.1f));
}

// ============================================================================
// Input Size Configuration Tests
// ============================================================================

TEST_F(YunetFaceDetectorTest, SetInputSize_AffectsDetection)
{
    ASSERT_TRUE(detector->loadModel(modelPath));

    // Test different input sizes
    detector->setInputSize(cv::Size(160, 160)); // Smaller input
    auto smallInputResults = detector->detectFaces(testImage);

    detector->setInputSize(cv::Size(640, 640)); // Larger input
    auto largeInputResults = detector->detectFaces(testImage);

    // Both should detect faces, but possibly different counts due to resolution
    EXPECT_FALSE(smallInputResults.empty() && largeInputResults.empty());

    for (const auto& result : smallInputResults)
    {
        EXPECT_TRUE(isValidDetection(result, testImage.size()));
    }
    for (const auto& result : largeInputResults)
    {
        EXPECT_TRUE(isValidDetection(result, testImage.size()));
    }
}

TEST_F(YunetFaceDetectorTest, SetInputSize_VerySmall_StillWorks)
{
    ASSERT_TRUE(detector->loadModel(modelPath));

    detector->setInputSize(cv::Size(64, 64));

    EXPECT_NO_THROW({
        auto results = detector->detectFaces(testImage);
        // Might not detect faces due to very low resolution, but shouldn't
        // crash
    });
}

// ============================================================================
// Parameter Integration Tests
// ============================================================================

TEST_F(YunetFaceDetectorTest, SetDetectionParams_InterfaceCompliance)
{
    ASSERT_TRUE(detector->loadModel(modelPath));

    // Test the interface compliance (scaleFactor and minNeighbors ignored for
    // DNN)
    detector->setDetectionParams(1.5, 10, cv::Size(40, 40), cv::Size(200, 200));

    auto results = detector->detectFaces(testImage);

    // Should still work (DNN ignores scaleFactor and minNeighbors)
    EXPECT_FALSE(results.empty());

    // Check that min/max size constraints are respected
    for (const auto& result : results)
    {
        EXPECT_GE(result.boundingBox.width, 40);
        EXPECT_GE(result.boundingBox.height, 40);
        EXPECT_LE(result.boundingBox.width, 200);
        EXPECT_LE(result.boundingBox.height, 200);
    }
}

// ============================================================================
// Interface Compliance Tests
// ============================================================================

TEST_F(YunetFaceDetectorTest, GetSupportedExtensions_ReturnsONNX)
{
    auto extensions = detector->getSupportedExtensions();

    EXPECT_FALSE(extensions.empty());
    bool hasOnnxExtension =
      std::find(extensions.begin(), extensions.end(), ".onnx") !=
      extensions.end();
    EXPECT_TRUE(hasOnnxExtension);
}

TEST_F(YunetFaceDetectorTest, GetModelInfo_BeforeLoading_ReturnsEmptyOrDefault)
{
    std::string info = detector->getModelInfo();

    EXPECT_TRUE(
      info.empty() || info.find("No model") != std::string::npos ||
      info.find("not loaded") != std::string::npos
    );
}

TEST_F(YunetFaceDetectorTest, GetModelInfo_AfterLoading_ContainsModelInfo)
{
    ASSERT_TRUE(detector->loadModel(modelPath));

    std::string info = detector->getModelInfo();

    EXPECT_FALSE(info.empty());
    EXPECT_TRUE(
      info.find("yunet") != std::string::npos ||
      info.find("loaded") != std::string::npos ||
      info.find("onnx") != std::string::npos ||
      info.find("DNN") != std::string::npos
    );
}

// ============================================================================
// Performance and Robustness Tests
// ============================================================================

TEST_F(
  YunetFaceDetectorTest, DetectFaces_MultipleCallsSameImage_ConsistentResults
)
{
    ASSERT_TRUE(detector->loadModel(modelPath));

    auto results1 = detector->detectFaces(testImage);
    auto results2 = detector->detectFaces(testImage);
    auto results3 = detector->detectFaces(testImage);

    EXPECT_EQ(results1.size(), results2.size());
    EXPECT_EQ(results2.size(), results3.size());

    // Results should be very similar (may have small floating point
    // differences)
    for (size_t i = 0; i < std::min(results1.size(), results2.size()); ++i)
    {
        cv::Rect box1 = results1[i].boundingBox;
        cv::Rect box2 = results2[i].boundingBox;

        // Allow small differences due to floating point precision
        EXPECT_NEAR(box1.x, box2.x, 2);
        EXPECT_NEAR(box1.y, box2.y, 2);
        EXPECT_NEAR(box1.width, box2.width, 2);
        EXPECT_NEAR(box1.height, box2.height, 2);
        EXPECT_NEAR(results1[i].confidence, results2[i].confidence, 0.01f);
    }
}

TEST_F(YunetFaceDetectorTest, DetectFaces_VerySmallImage_HandledGracefully)
{
    ASSERT_TRUE(detector->loadModel(modelPath));

    cv::Mat smallImage = cv::Mat::zeros(32, 32, CV_8UC3);

    EXPECT_NO_THROW({
        auto results = detector->detectFaces(smallImage);
        // Might not detect faces in very small image, but shouldn't crash
    });
}

TEST_F(YunetFaceDetectorTest, DetectFaces_VeryLargeImage_HandledCorrectly)
{
    ASSERT_TRUE(detector->loadModel(modelPath));

    cv::Mat largeImage;
    cv::resize(testImage, largeImage, cv::Size(1920, 1080));

    EXPECT_NO_THROW({
        auto results = detector->detectFaces(largeImage);
        // Should handle large images without issues
        for (const auto& result : results)
        {
            EXPECT_TRUE(isValidDetection(result, largeImage.size()));
        }
    });
}

// ============================================================================
// Move Semantics Tests
// ============================================================================

TEST_F(YunetFaceDetectorTest, MoveConstructor_WorksCorrectly)
{
    ASSERT_TRUE(detector->loadModel(modelPath));
    ASSERT_TRUE(detector->isLoaded());

    // Move construct new detector
    YunetFaceDetector movedDetector = std::move(*detector);

    EXPECT_TRUE(movedDetector.isLoaded());
    EXPECT_FALSE(movedDetector.getModelInfo().empty());

    auto results = movedDetector.detectFaces(testImage);
    EXPECT_TRUE(isExpectedFaceCount(
      results.size(), EXPECTED_FACE_COUNT, FACE_COUNT_TOLERANCE
    ));
}

// ============================================================================
// Edge Cases and Error Handling
// ============================================================================

TEST_F(YunetFaceDetectorTest, DetectFaces_CorruptedImage_HandledGracefully)
{
    ASSERT_TRUE(detector->loadModel(modelPath));

    cv::Mat corruptedImage =
      cv::Mat::ones(100, 100, CV_32F) * 1000; // Invalid pixel values

    EXPECT_NO_THROW({
        auto results = detector->detectFaces(corruptedImage);
        // Should handle gracefully, possibly return empty results
    });
}

TEST_F(YunetFaceDetectorTest, DetectFaces_SinglePixelImage_HandledGracefully)
{
    ASSERT_TRUE(detector->loadModel(modelPath));

    cv::Mat singlePixel = cv::Mat::ones(1, 1, CV_8UC3);

    EXPECT_NO_THROW({
        auto results = detector->detectFaces(singlePixel);
        EXPECT_TRUE(results.empty()); // Cannot detect faces in 1x1 image
    });
}

/**
 * @brief Test CUDA acceleration functionality
 *
 * Tests CUDA availability detection, enable/disable functionality, and
 * performance with both CPU and GPU backends when available.
 */
TEST_F(YunetFaceDetectorTest, CudaAcceleration)
{
    // Test CUDA availability detection
    bool cudaAvailable = YunetFaceDetector::isCudaAvailable();
    std::cout << "CUDA available on test system: "
              << (cudaAvailable ? "Yes" : "No") << std::endl;

    if (!cudaAvailable)
    {
        GTEST_SKIP() << "CUDA not available on this system";
        return;
    }

    // Load model for CUDA testing
    ASSERT_TRUE(detector->loadModel(modelPath))
      << "Failed to load model for CUDA testing";

    // Test initial CUDA state (should be enabled by default if available)
    EXPECT_TRUE(detector->isCudaEnabled())
      << "CUDA should be enabled by default when available";

    // Test disabling CUDA
    EXPECT_TRUE(detector->setCudaEnabled(false))
      << "Should be able to disable CUDA";
    EXPECT_FALSE(detector->isCudaEnabled())
      << "CUDA should be disabled after setCudaEnabled(false)";

    // Test re-enabling CUDA
    EXPECT_TRUE(detector->setCudaEnabled(true))
      << "Should be able to re-enable CUDA";
    EXPECT_TRUE(detector->isCudaEnabled())
      << "CUDA should be enabled after setCudaEnabled(true)";

    // Test model info includes CUDA status
    std::string modelInfo = detector->getModelInfo();
    EXPECT_TRUE(modelInfo.find("CUDA") != std::string::npos)
      << "Model info should include CUDA status: " << modelInfo;

    // Basic functionality test with CUDA enabled
    auto cudaResults = detector->detectFaces(testImage);
    EXPECT_FALSE(cudaResults.empty())
      << "Should detect faces with CUDA enabled";

    // Test with CUDA disabled for comparison
    detector->setCudaEnabled(false);
    auto cpuResults = detector->detectFaces(testImage);
    EXPECT_FALSE(cpuResults.empty())
      << "Should detect faces with CUDA disabled (CPU fallback)";

    // Results should be consistent between CUDA and CPU (within tolerance)
    EXPECT_EQ(cudaResults.size(), cpuResults.size())
      << "CUDA and CPU should detect the same number of faces";

    // Re-enable CUDA for cleanup
    detector->setCudaEnabled(true);
}

/**
 * @brief Test CUDA error handling and fallback behavior
 */
TEST_F(YunetFaceDetectorTest, CudaErrorHandling)
{
    if (!YunetFaceDetector::isCudaAvailable())
    {
        GTEST_SKIP() << "CUDA not available for error handling test";
    }

    // Create detector and verify CUDA works initially
    auto cudaDetector = std::make_unique<YunetFaceDetector>();
    EXPECT_TRUE(cudaDetector->isCudaEnabled())
      << "CUDA should be enabled by default";

    // Load model
    ASSERT_TRUE(cudaDetector->loadModel(modelPath));

    // Test that setting unavailable CUDA fails gracefully
    // (This is a hypothetical test - in practice, CUDA should work if
    // available) The main point is ensuring no crashes occur

    // Verify basic operations work regardless of backend
    EXPECT_NO_THROW({
        auto results = cudaDetector->detectFaces(testImage);
        EXPECT_FALSE(results.empty());
    });

    // Test model info is properly updated
    std::string info = cudaDetector->getModelInfo();
    EXPECT_FALSE(info.empty());
    EXPECT_TRUE(info.find("DNN Face Detector") != std::string::npos);
}
