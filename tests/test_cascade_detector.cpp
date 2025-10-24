// Copyright (c) 2025 JackLee
//
// This software is released under the MIT License.
// https://opensource.org/licenses/MIT

#include <gtest/gtest.h>
#include <opencv2/opencv.hpp>
#include <filesystem>
#include <memory>
#include "CascadeFaceDetector.h"

/**
 * @brief Test fixture for CascadeFaceDetector testing
 *
 * This test fixture provides common setup and utilities for all cascade
 * detector tests. It handles test data paths, image loading, and validation
 * helpers.
 */
class CascadeFaceDetectorTest : public ::testing::Test
{
  protected:
    void SetUp() override
    {
        // Setup test data paths
        modelPath = "models/haarcascade_frontalface_alt.xml";
        testImagePath = "test_images/test01.jpg";
        invalidModelPath = "models/nonexistent.xml";

        // Create detector instance
        detector = std::make_unique<CascadeFaceDetector>();

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

  protected:
    std::unique_ptr<CascadeFaceDetector> detector;
    std::string modelPath;
    std::string testImagePath;
    std::string invalidModelPath;
    cv::Mat testImage;

    // Expected results for test01.jpg
    static constexpr size_t EXPECTED_FACE_COUNT = 3;
    static constexpr size_t FACE_COUNT_TOLERANCE = 1;
};

// ============================================================================
// Model Loading Tests
// ============================================================================

TEST_F(CascadeFaceDetectorTest, LoadValidModel_Success)
{
    bool result = detector->loadModel(modelPath);

    EXPECT_TRUE(result);
    EXPECT_TRUE(detector->isLoaded());
    EXPECT_FALSE(detector->getModelInfo().empty());
    std::string modelInfo = detector->getModelInfo();
    EXPECT_TRUE(
      modelInfo.find("haarcascade") != std::string::npos ||
      modelInfo.find("loaded") != std::string::npos
    );
}

TEST_F(CascadeFaceDetectorTest, LoadInvalidModel_Failure)
{
    bool result = detector->loadModel(invalidModelPath);

    EXPECT_FALSE(result);
    EXPECT_FALSE(detector->isLoaded());
}

TEST_F(CascadeFaceDetectorTest, LoadEmptyPath_Failure)
{
    bool result = detector->loadModel("");

    EXPECT_FALSE(result);
    EXPECT_FALSE(detector->isLoaded());
}

TEST_F(CascadeFaceDetectorTest, LoadModelTwice_SecondLoadSucceeds)
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

TEST_F(CascadeFaceDetectorTest, DetectFaces_ValidImage_DetectsExpectedFaces)
{
    ASSERT_TRUE(detector->loadModel(modelPath));

    auto results = detector->detectFaces(testImage);

    EXPECT_TRUE(isExpectedFaceCount(
      results.size(), EXPECTED_FACE_COUNT, FACE_COUNT_TOLERANCE
    ));

    // Validate all detections
    for (const auto& result : results)
    {
        EXPECT_TRUE(isValidDetection(result, testImage.size()));
        EXPECT_EQ(
          result.confidence, 1.0f
        ); // Cascade detectors return confidence 1.0
    }
}

TEST_F(CascadeFaceDetectorTest, DetectFaces_EmptyImage_ReturnsEmpty)
{
    ASSERT_TRUE(detector->loadModel(modelPath));

    cv::Mat emptyImage;
    auto results = detector->detectFaces(emptyImage);

    EXPECT_TRUE(results.empty());
}

TEST_F(CascadeFaceDetectorTest, DetectFaces_NoModelLoaded_ReturnsEmpty)
{
    // Don't load model
    ASSERT_FALSE(detector->isLoaded());

    auto results = detector->detectFaces(testImage);

    EXPECT_TRUE(results.empty());
}

TEST_F(CascadeFaceDetectorTest, DetectFaces_GrayscaleImage_WorksCorrectly)
{
    ASSERT_TRUE(detector->loadModel(modelPath));

    cv::Mat grayImage;
    cv::cvtColor(testImage, grayImage, cv::COLOR_BGR2GRAY);

    auto results = detector->detectFaces(grayImage);

    EXPECT_TRUE(isExpectedFaceCount(
      results.size(), EXPECTED_FACE_COUNT, FACE_COUNT_TOLERANCE
    ));
}

TEST_F(CascadeFaceDetectorTest, DetectFaces_SingleChannelImage_HandledCorrectly)
{
    ASSERT_TRUE(detector->loadModel(modelPath));

    cv::Mat grayImage;
    cv::cvtColor(testImage, grayImage, cv::COLOR_BGR2GRAY);

    auto results = detector->detectFaces(grayImage);

    EXPECT_FALSE(results.empty());
    for (const auto& result : results)
    {
        EXPECT_TRUE(isValidDetection(result, grayImage.size()));
    }
}

// ============================================================================
// Parameter Configuration Tests
// ============================================================================

TEST_F(CascadeFaceDetectorTest, SetDetectionParams_AffectsDetection)
{
    ASSERT_TRUE(detector->loadModel(modelPath));

    // Test with default parameters
    auto defaultResults = detector->detectFaces(testImage);

    // Test with more strict parameters (should detect fewer faces)
    detector->setDetectionParams(1.3, 5, cv::Size(50, 50));
    auto strictResults = detector->detectFaces(testImage);

    // Test with more lenient parameters (might detect more faces)
    detector->setDetectionParams(1.05, 2, cv::Size(20, 20));
    auto lenientResults = detector->detectFaces(testImage);

    // Validate that different parameters produce different results
    EXPECT_TRUE(
      defaultResults.size() >= strictResults.size() ||
      defaultResults.size() <= lenientResults.size()
    );
}

TEST_F(
  CascadeFaceDetectorTest, SetDetectionParams_InvalidParams_HandledGracefully
)
{
    ASSERT_TRUE(detector->loadModel(modelPath));

    // Test with invalid scale factor (should be > 1.0)
    detector->setDetectionParams(0.5, 3, cv::Size(30, 30));
    auto results = detector->detectFaces(testImage);

    // Should still work (OpenCV handles invalid params internally)
    EXPECT_NO_THROW(detector->detectFaces(testImage));
}

TEST_F(
  CascadeFaceDetectorTest,
  SetDetectionParams_NegativeMinNeighbors_HandledGracefully
)
{
    ASSERT_TRUE(detector->loadModel(modelPath));

    detector->setDetectionParams(1.1, -1, cv::Size(30, 30));

    EXPECT_NO_THROW({ auto results = detector->detectFaces(testImage); });
}

// ============================================================================
// Interface Compliance Tests
// ============================================================================

TEST_F(CascadeFaceDetectorTest, GetSupportedExtensions_ReturnsXML)
{
    auto extensions = detector->getSupportedExtensions();

    EXPECT_FALSE(extensions.empty());
    bool hasXmlExtension =
      std::find(extensions.begin(), extensions.end(), ".xml") !=
      extensions.end();
    EXPECT_TRUE(hasXmlExtension);
}

TEST_F(
  CascadeFaceDetectorTest, GetModelInfo_BeforeLoading_ReturnsEmptyOrDefault
)
{
    std::string info = detector->getModelInfo();

    EXPECT_TRUE(
      info.empty() || info.find("No model") != std::string::npos ||
      info.find("not loaded") != std::string::npos
    );
}

TEST_F(CascadeFaceDetectorTest, GetModelInfo_AfterLoading_ContainsModelInfo)
{
    ASSERT_TRUE(detector->loadModel(modelPath));

    std::string info = detector->getModelInfo();

    EXPECT_FALSE(info.empty());
    EXPECT_TRUE(
      info.find("haarcascade") != std::string::npos ||
      info.find("loaded") != std::string::npos ||
      info.find("xml") != std::string::npos
    );
}

// ============================================================================
// Performance and Robustness Tests
// ============================================================================

TEST_F(
  CascadeFaceDetectorTest, DetectFaces_MultipleCallsSameImage_ConsistentResults
)
{
    ASSERT_TRUE(detector->loadModel(modelPath));

    auto results1 = detector->detectFaces(testImage);
    auto results2 = detector->detectFaces(testImage);
    auto results3 = detector->detectFaces(testImage);

    // Count should be consistent
    EXPECT_EQ(results1.size(), results2.size());
    EXPECT_EQ(results2.size(), results3.size());

    // Results should contain the same detections (but order may vary)
    // Sort detections by area for comparison
    auto sortByArea = [](const DetectionResult& a, const DetectionResult& b)
    {
        return a.boundingBox.area() < b.boundingBox.area();
    };

    std::sort(results1.begin(), results1.end(), sortByArea);
    std::sort(results2.begin(), results2.end(), sortByArea);

    // Now compare sorted results (should be identical)
    for (size_t i = 0; i < results1.size() && i < results2.size(); ++i)
    {
        // Allow small coordinate differences due to internal processing
        // variations
        const auto& box1 = results1[i].boundingBox;
        const auto& box2 = results2[i].boundingBox;

        EXPECT_NEAR(box1.x, box2.x, 2)
          << "X coordinate should be nearly identical";
        EXPECT_NEAR(box1.y, box2.y, 2)
          << "Y coordinate should be nearly identical";
        EXPECT_NEAR(box1.width, box2.width, 2)
          << "Width should be nearly identical";
        EXPECT_NEAR(box1.height, box2.height, 2)
          << "Height should be nearly identical";
        EXPECT_EQ(results1[i].confidence, results2[i].confidence);
    }
}

TEST_F(CascadeFaceDetectorTest, DetectFaces_VerySmallImage_HandledGracefully)
{
    ASSERT_TRUE(detector->loadModel(modelPath));

    cv::Mat smallImage = cv::Mat::zeros(10, 10, CV_8UC3);

    EXPECT_NO_THROW({
        auto results = detector->detectFaces(smallImage);
        EXPECT_TRUE(results.empty()); // No faces in 10x10 image
    });
}

TEST_F(CascadeFaceDetectorTest, DetectFaces_VeryLargeMinSize_ReturnsEmpty)
{
    ASSERT_TRUE(detector->loadModel(modelPath));

    // Set minimum size larger than image
    detector->setDetectionParams(1.1, 3, cv::Size(2000, 2000));
    auto results = detector->detectFaces(testImage);

    EXPECT_TRUE(results.empty());
}

// ============================================================================
// Move Semantics Tests
// ============================================================================

TEST_F(CascadeFaceDetectorTest, MoveConstructor_WorksCorrectly)
{
    ASSERT_TRUE(detector->loadModel(modelPath));
    ASSERT_TRUE(detector->isLoaded());

    // Move construct new detector
    CascadeFaceDetector movedDetector = std::move(*detector);

    EXPECT_TRUE(movedDetector.isLoaded());
    EXPECT_FALSE(movedDetector.getModelInfo().empty());

    auto results = movedDetector.detectFaces(testImage);
    EXPECT_TRUE(isExpectedFaceCount(
      results.size(), EXPECTED_FACE_COUNT, FACE_COUNT_TOLERANCE
    ));
}

// ============================================================================
// CUDA Acceleration Tests
// ============================================================================

/**
 * @brief Test CUDA acceleration functionality for Cascade detector
 *
 * Tests CUDA availability detection, enable/disable functionality, and
 * performance with both CPU and GPU backends when available.
 * Note: CascadeFaceDetector uses GPU for preprocessing only (grayscale
 * conversion and histogram equalization), with actual cascade detection on CPU.
 */
TEST_F(CascadeFaceDetectorTest, CudaAcceleration)
{
    // Test CUDA availability detection
    bool cudaAvailable = CascadeFaceDetector::isCudaAvailable();
    std::cout << "CUDA available for Cascade detector: "
              << (cudaAvailable ? "Yes" : "No") << std::endl;

    if (!cudaAvailable)
    {
        GTEST_SKIP()
          << "CUDA not available on this system for Cascade detector";
        return;
    }

    // Load model for CUDA testing
    ASSERT_TRUE(detector->loadModel(modelPath))
      << "Failed to load model for CUDA testing";

    // Test initial CUDA state (should be disabled by default)
    EXPECT_FALSE(detector->isCudaEnabled())
      << "CUDA should be disabled by default for Cascade detector";

    // Test enabling CUDA
    EXPECT_TRUE(detector->setCudaEnabled(true))
      << "Should be able to enable CUDA";
    EXPECT_TRUE(detector->isCudaEnabled())
      << "CUDA should be enabled after setCudaEnabled(true)";

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
      << "Should detect faces with CUDA disabled (CPU only)";

    // Results should be consistent between CUDA preprocessing and CPU-only
    // (allowing small variations due to different preprocessing paths)
    EXPECT_TRUE(isExpectedFaceCount(cudaResults.size(), cpuResults.size(), 1))
      << "CUDA and CPU should detect similar number of faces (within "
         "tolerance)";

    // Re-enable CUDA for cleanup
    detector->setCudaEnabled(true);
}

/**
 * @brief Test CUDA error handling and fallback behavior for Cascade detector
 */
TEST_F(CascadeFaceDetectorTest, CudaErrorHandling)
{
    if (!CascadeFaceDetector::isCudaAvailable())
    {
        GTEST_SKIP() << "CUDA not available for error handling test";
    }

    // Create detector and test CUDA controls
    auto cudaDetector = std::make_unique<CascadeFaceDetector>();

    // Initial state should be CUDA disabled
    EXPECT_FALSE(cudaDetector->isCudaEnabled())
      << "CUDA should be disabled by default";

    // Load model
    ASSERT_TRUE(cudaDetector->loadModel(modelPath));

    // Test enabling CUDA
    EXPECT_TRUE(cudaDetector->setCudaEnabled(true));

    // Verify basic operations work with CUDA enabled
    EXPECT_NO_THROW({
        auto results = cudaDetector->detectFaces(testImage);
        EXPECT_FALSE(results.empty());
    });

    // Test that fallback works (disabling CUDA mid-operation)
    cudaDetector->setCudaEnabled(false);
    EXPECT_NO_THROW({
        auto results = cudaDetector->detectFaces(testImage);
        EXPECT_FALSE(results.empty());
    });

    // Test model info is properly updated
    std::string info = cudaDetector->getModelInfo();
    EXPECT_FALSE(info.empty());
    EXPECT_TRUE(info.find("Haar Cascade") != std::string::npos);
}

/**
 * @brief Test CUDA performance characteristics
 */
TEST_F(CascadeFaceDetectorTest, CudaPerformanceCharacteristics)
{
    if (!CascadeFaceDetector::isCudaAvailable())
    {
        GTEST_SKIP() << "CUDA not available for performance test";
    }

    ASSERT_TRUE(detector->loadModel(modelPath));

    // Test with multiple image sizes
    std::vector<cv::Size> testSizes = {
      cv::Size(320, 240), // Small
      cv::Size(640, 480), // Medium
      cv::Size(1280, 720) // Large
    };

    for (const auto& size : testSizes)
    {
        cv::Mat resizedImage;
        cv::resize(testImage, resizedImage, size);

        // Test CUDA preprocessing
        detector->setCudaEnabled(true);
        EXPECT_NO_THROW({
            auto cudaResults = detector->detectFaces(resizedImage);
            // Should work for all sizes
        });

        // Test CPU-only
        detector->setCudaEnabled(false);
        EXPECT_NO_THROW({
            auto cpuResults = detector->detectFaces(resizedImage);
            // Should work for all sizes
        });
    }
}

/**
 * @brief Test CUDA availability static method consistency
 */
TEST_F(CascadeFaceDetectorTest, CudaAvailabilityConsistency)
{
    // Static method should return consistent results
    bool available1 = CascadeFaceDetector::isCudaAvailable();
    bool available2 = CascadeFaceDetector::isCudaAvailable();
    bool available3 = CascadeFaceDetector::isCudaAvailable();

    EXPECT_EQ(available1, available2);
    EXPECT_EQ(available2, available3);

    std::cout << "CUDA availability check consistency: "
              << (available1 ? "Available" : "Not Available") << std::endl;
}
