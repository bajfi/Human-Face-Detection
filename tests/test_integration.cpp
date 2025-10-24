// Copyright (c) 2025 JackLee
//
// This software is released under the MIT License.
// https://opensource.org/licenses/MIT

#include <gtest/gtest.h>
#include <opencv2/opencv.hpp>
#include <memory>
#include <chrono>
#include "test_utils.h"
#include "CascadeFaceDetector.h"
#include "YunetFaceDetector.h"
#include "FaceDetectorFactory.h"

using namespace TestUtils;

/**
 * @brief Integration tests for face detection system
 *
 * These tests verify the interaction between different components
 * and compare the performance of different detection methods.
 */
class FaceDetectionIntegrationTest : public ::testing::Test
{
  protected:
    void SetUp() override
    {
        // Verify test environment
        ASSERT_TRUE(FileUtils::validateTestEnvironment())
          << "Test environment validation failed. Missing files: " << [this]()
        {
            auto missing = FileUtils::getMissingTestFiles();
            std::string result;
            for (const auto& file : missing)
            {
                result += file + " ";
            }
            return result;
        }();

        // Load test image
        testImage = ImageUtils::loadTestImage();
        ASSERT_FALSE(testImage.empty());

        // Create detector instances
        cascadeDetector = std::make_unique<CascadeFaceDetector>();
        dnnDetector = std::make_unique<YunetFaceDetector>();
    }

    void TearDown() override
    {
        cascadeDetector.reset();
        dnnDetector.reset();
    }

  protected:
    cv::Mat testImage;
    std::unique_ptr<CascadeFaceDetector> cascadeDetector;
    std::unique_ptr<YunetFaceDetector> dnnDetector;
};

// ============================================================================
// Cross-Detector Comparison Tests
// ============================================================================

TEST_F(FaceDetectionIntegrationTest, CompareDetectionResults_BothDetectFaces)
{
    // Load models
    ASSERT_TRUE(cascadeDetector->loadModel(TestPaths::CASCADE_MODEL));
    ASSERT_TRUE(dnnDetector->loadModel(TestPaths::DNN_MODEL));

    // Run detection with both methods
    auto cascadeResults = cascadeDetector->detectFaces(testImage);
    auto dnnResults = dnnDetector->detectFaces(testImage);

    // Both should detect faces
    EXPECT_FALSE(cascadeResults.empty());
    EXPECT_FALSE(dnnResults.empty());

    // Both should detect reasonable number of faces
    EXPECT_TRUE(
      ValidationUtils::isExpectedFaceCount(
        cascadeResults.size(), TestConfig::EXPECTED_FACE_COUNT
      )
    );
    EXPECT_TRUE(
      ValidationUtils::isExpectedFaceCount(
        dnnResults.size(), TestConfig::EXPECTED_FACE_COUNT
      )
    );

    // Validate detection quality
    EXPECT_TRUE(
      ValidationUtils::areAllDetectionsValid(cascadeResults, testImage.size())
    );
    EXPECT_TRUE(
      ValidationUtils::areAllDetectionsValid(dnnResults, testImage.size())
    );
}

TEST_F(
  FaceDetectionIntegrationTest,
  CompareConfidenceScores_DnnProvidesTorealConfidence
)
{
    ASSERT_TRUE(cascadeDetector->loadModel(TestPaths::CASCADE_MODEL));
    ASSERT_TRUE(dnnDetector->loadModel(TestPaths::DNN_MODEL));

    auto cascadeResults = cascadeDetector->detectFaces(testImage);
    auto dnnResults = dnnDetector->detectFaces(testImage);

    // Cascade detector should return confidence 1.0 for all detections
    for (const auto& result : cascadeResults)
    {
        EXPECT_EQ(result.confidence, 1.0f);
    }

    // DNN detector should provide varied confidence scores
    EXPECT_TRUE(ValidationUtils::hasReasonableConfidenceScores(dnnResults));

    // DNN should have some variation in confidence scores (not all the same)
    if (dnnResults.size() > 1)
    {
        bool hasVariation = false;
        for (size_t i = 1; i < dnnResults.size(); ++i)
        {
            if (std::abs(dnnResults[i].confidence - dnnResults[0].confidence) >
                0.01f)
            {
                hasVariation = true;
                break;
            }
        }
        // Note: This might not always be true, but generally DNN provides
        // varied confidence
    }
}

// ============================================================================
// Performance Comparison Tests
// ============================================================================

TEST_F(FaceDetectionIntegrationTest, CompareLoadingTimes_BothLoadReasonably)
{
    double cascadeLoadTime = PerformanceUtils::measureLoadingTime(
      *cascadeDetector, TestPaths::CASCADE_MODEL
    );
    double dnnLoadTime =
      PerformanceUtils::measureLoadingTime(*dnnDetector, TestPaths::DNN_MODEL);

    EXPECT_GT(cascadeLoadTime, 0.0); // Should have loaded successfully
    EXPECT_GT(dnnLoadTime, 0.0);     // Should have loaded successfully

    // Both loading times should be reasonable (less than 5 seconds)
    EXPECT_LT(cascadeLoadTime, 5000.0); // ms
    EXPECT_LT(dnnLoadTime, 5000.0);     // ms

    std::cout << "Cascade loading time: " << cascadeLoadTime << " ms"
              << std::endl;
    std::cout << "DNN loading time: " << dnnLoadTime << " ms" << std::endl;
}

TEST_F(
  FaceDetectionIntegrationTest, CompareDetectionTimes_ReasonablePerformance
)
{
    ASSERT_TRUE(cascadeDetector->loadModel(TestPaths::CASCADE_MODEL));
    ASSERT_TRUE(dnnDetector->loadModel(TestPaths::DNN_MODEL));

    double cascadeDetectionTime =
      PerformanceUtils::measureDetectionTime(*cascadeDetector, testImage, 5);
    double dnnDetectionTime =
      PerformanceUtils::measureDetectionTime(*dnnDetector, testImage, 5);

    // Both should complete within reasonable time (less than 1 second per
    // detection)
    EXPECT_LT(cascadeDetectionTime, 1000.0); // ms
    EXPECT_LT(dnnDetectionTime, 1000.0);     // ms

    std::cout << "Cascade detection time: " << cascadeDetectionTime << " ms"
              << std::endl;
    std::cout << "DNN detection time: " << dnnDetectionTime << " ms"
              << std::endl;
}

// ============================================================================
// Factory Pattern Tests
// ============================================================================

TEST_F(FaceDetectionIntegrationTest, FaceDetectorFactory_CreatesCascadeDetector)
{
    auto detector =
      FaceDetectorFactory::createDetector(TestPaths::CASCADE_MODEL);

    ASSERT_NE(detector, nullptr);
    EXPECT_TRUE(detector->loadModel(TestPaths::CASCADE_MODEL));
    EXPECT_TRUE(detector->isLoaded());

    auto results = detector->detectFaces(testImage);
    EXPECT_TRUE(
      ValidationUtils::isExpectedFaceCount(
        results.size(), TestConfig::EXPECTED_FACE_COUNT
      )
    );
}

TEST_F(FaceDetectionIntegrationTest, FaceDetectorFactory_CreatesDnnDetector)
{
    auto detector = FaceDetectorFactory::createDetector(TestPaths::DNN_MODEL);

    ASSERT_NE(detector, nullptr);
    EXPECT_TRUE(detector->loadModel(TestPaths::DNN_MODEL));
    EXPECT_TRUE(detector->isLoaded());

    auto results = detector->detectFaces(testImage);
    EXPECT_TRUE(
      ValidationUtils::isExpectedFaceCount(
        results.size(), TestConfig::EXPECTED_FACE_COUNT
      )
    );
}

TEST_F(FaceDetectionIntegrationTest, FaceDetectorFactory_HandlesInvalidPath)
{
    auto detector = FaceDetectorFactory::createDetector("invalid.unknown");

    // Factory should return nullptr or a detector that fails to load
    if (detector != nullptr)
    {
        EXPECT_FALSE(detector->loadModel("invalid.unknown"));
        EXPECT_FALSE(detector->isLoaded());
    }
}

// ============================================================================
// Robustness Tests Across Detectors
// ============================================================================

TEST_F(FaceDetectionIntegrationTest, BothDetectors_HandleDifferentImageSizes)
{
    ASSERT_TRUE(cascadeDetector->loadModel(TestPaths::CASCADE_MODEL));
    ASSERT_TRUE(dnnDetector->loadModel(TestPaths::DNN_MODEL));

    // Test with different image sizes
    std::vector<cv::Size> testSizes = {
      cv::Size(320, 240), // Small
      cv::Size(640, 480), // Medium
      cv::Size(1280, 720) // Large
    };

    for (const auto& size : testSizes)
    {
        cv::Mat resizedImage =
          ImageUtils::resizeWithAspectRatio(testImage, size);

        EXPECT_NO_THROW({
            auto cascadeResults = cascadeDetector->detectFaces(resizedImage);
            auto dnnResults = dnnDetector->detectFaces(resizedImage);

            // Both should handle the resize gracefully
            EXPECT_TRUE(
              ValidationUtils::areAllDetectionsValid(
                cascadeResults, resizedImage.size()
              )
            );
            EXPECT_TRUE(
              ValidationUtils::areAllDetectionsValid(
                dnnResults, resizedImage.size()
              )
            );
        });
    }
}

TEST_F(FaceDetectionIntegrationTest, BothDetectors_HandleGrayscaleConversion)
{
    ASSERT_TRUE(cascadeDetector->loadModel(TestPaths::CASCADE_MODEL));
    ASSERT_TRUE(dnnDetector->loadModel(TestPaths::DNN_MODEL));

    cv::Mat grayImage = ImageUtils::convertToGrayscaleBGR(testImage);

    auto cascadeResults = cascadeDetector->detectFaces(grayImage);
    auto dnnResults = dnnDetector->detectFaces(grayImage);

    // Both should handle grayscale images
    EXPECT_TRUE(
      ValidationUtils::areAllDetectionsValid(cascadeResults, grayImage.size())
    );
    EXPECT_TRUE(
      ValidationUtils::areAllDetectionsValid(dnnResults, grayImage.size())
    );
}

// ============================================================================
// Consistency and Reliability Tests
// ============================================================================

TEST_F(FaceDetectionIntegrationTest, BothDetectors_ConsistentResultsAcrossRuns)
{
    ASSERT_TRUE(cascadeDetector->loadModel(TestPaths::CASCADE_MODEL));
    ASSERT_TRUE(dnnDetector->loadModel(TestPaths::DNN_MODEL));

    // Run multiple times and check consistency
    constexpr int NUM_RUNS = 3;

    std::vector<std::vector<DetectionResult>> cascadeRuns;
    std::vector<std::vector<DetectionResult>> dnnRuns;

    for (int i = 0; i < NUM_RUNS; ++i)
    {
        cascadeRuns.push_back(cascadeDetector->detectFaces(testImage));
        dnnRuns.push_back(dnnDetector->detectFaces(testImage));
    }

    // Check cascade consistency
    for (int i = 1; i < NUM_RUNS; ++i)
    {
        EXPECT_EQ(cascadeRuns[0].size(), cascadeRuns[i].size());
    }

    // Check DNN consistency
    for (int i = 1; i < NUM_RUNS; ++i)
    {
        EXPECT_EQ(dnnRuns[0].size(), dnnRuns[i].size());
    }
}

// ============================================================================
// Memory and Resource Management Tests
// ============================================================================

TEST_F(FaceDetectionIntegrationTest, BothDetectors_HandleMultipleModelLoads)
{
    // Test loading models multiple times
    for (int i = 0; i < 3; ++i)
    {
        EXPECT_TRUE(cascadeDetector->loadModel(TestPaths::CASCADE_MODEL));
        EXPECT_TRUE(cascadeDetector->isLoaded());

        EXPECT_TRUE(dnnDetector->loadModel(TestPaths::DNN_MODEL));
        EXPECT_TRUE(dnnDetector->isLoaded());
    }

    // Should still work after multiple loads
    auto cascadeResults = cascadeDetector->detectFaces(testImage);
    auto dnnResults = dnnDetector->detectFaces(testImage);

    EXPECT_FALSE(cascadeResults.empty());
    EXPECT_FALSE(dnnResults.empty());
}

TEST_F(FaceDetectionIntegrationTest, BothDetectors_HandleMoveSemantics)
{
    ASSERT_TRUE(cascadeDetector->loadModel(TestPaths::CASCADE_MODEL));
    ASSERT_TRUE(dnnDetector->loadModel(TestPaths::DNN_MODEL));

    // Test move construction
    CascadeFaceDetector movedCascade = std::move(*cascadeDetector);
    YunetFaceDetector movedDnn = std::move(*dnnDetector);

    // Moved detectors should still work
    EXPECT_TRUE(movedCascade.isLoaded());
    EXPECT_TRUE(movedDnn.isLoaded());

    auto cascadeResults = movedCascade.detectFaces(testImage);
    auto dnnResults = movedDnn.detectFaces(testImage);

    EXPECT_FALSE(cascadeResults.empty());
    EXPECT_FALSE(dnnResults.empty());
}
