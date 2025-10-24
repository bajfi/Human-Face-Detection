// Copyright (c) 2025 JackLee
//
// This software is released under the MIT License.
// https://opensource.org/licenses/MIT

#ifndef FACEDETECTORFACTORY_H
#define FACEDETECTORFACTORY_H

#include "IFaceDetector.h"
#include "CascadeFaceDetector.h"
#include "YunetFaceDetector.h"
#include <string>
#include <memory>
#include <vector>

/**
 * @brief Factory class for creating appropriate face detector instances
 *
 * This factory implements the Factory Pattern to provide intelligent model
 * selection based on file extensions and model formats. It encapsulates the
 * logic for determining which detector implementation to use, making the system
 * easily extensible for future detector types.
 *
 * Design Benefits:
 * - Single point of detector creation and configuration
 * - Automatic model type detection based on file extension
 * - Easy extension for new detector types
 * - Centralized error handling and validation
 *
 * Usage:
 * ```cpp
 * auto detector = FaceDetectorFactory::createDetector("model.onnx");
 * if (detector && detector->loadModel("model.onnx")) {
 *     auto results = detector->detectFaces(image);
 * }
 * ```
 */
class FaceDetectorFactory
{
  public:
    /**
     * @brief Detector type enumeration for explicit type specification
     */
    enum class DetectorType
    {
        AUTO_DETECT, ///< Automatically detect based on file extension
        CASCADE,     ///< Haar Cascade Classifier
        DNN,         ///< Deep Neural Network (ONNX, TensorFlow, etc.)
        UNKNOWN      ///< Unknown or unsupported type
    };

    /**
     * @brief Model format information structure
     */
    struct ModelInfo
    {
        DetectorType type;
        std::string description;
        std::vector<std::string> supportedExtensions;
        bool isAvailable;

        ModelInfo(
          DetectorType t,
          const std::string& desc,
          const std::vector<std::string>& exts,
          bool available = true
        )
          : type(t),
            description(desc),
            supportedExtensions(exts),
            isAvailable(available)
        {}
    };

    /**
     * @brief Create face detector automatically based on model file extension
     * @param modelPath Path to the model file
     * @return Unique pointer to appropriate detector, or nullptr if unsupported
     */
    static FaceDetectorPtr createDetector(const std::string& modelPath);

    /**
     * @brief Create face detector of specific type
     * @param type Detector type to create
     * @return Unique pointer to detector, or nullptr if type not supported
     */
    static FaceDetectorPtr createDetector(DetectorType type);

    /**
     * @brief Determine detector type from model file extension
     * @param modelPath Path to the model file
     * @return Detected detector type or UNKNOWN if not supported
     */
    static DetectorType detectModelType(const std::string& modelPath);

    /**
     * @brief Check if a model file format is supported
     * @param modelPath Path to the model file
     * @return True if the file format is supported
     */
    static bool isModelSupported(const std::string& modelPath);

    /**
     * @brief Get information about all available detector types
     * @return Vector of ModelInfo structures describing available detectors
     */
    static std::vector<ModelInfo> getAvailableDetectors();

    /**
     * @brief Get supported file extensions for all detectors
     * @return Vector of all supported file extensions
     */
    static std::vector<std::string> getAllSupportedExtensions();

    /**
     * @brief Create detector with optimal default parameters
     * @param modelPath Path to model file
     * @return Configured detector ready for use, or nullptr on failure
     */
    static FaceDetectorPtr createOptimizedDetector(
      const std::string& modelPath
    );

    /**
     * @brief Get recommended detector type for performance vs accuracy
     * trade-off
     * @param prioritizeSpeed If true, recommends faster detectors; if false,
     * more accurate ones
     * @return Recommended detector type
     */
    static DetectorType getRecommendedDetector(bool prioritizeSpeed = false);

  private:
    /**
     * @brief Extract file extension from path (lowercase, with dot)
     * @param filePath Path to file
     * @return Lowercase extension including dot (e.g., ".onnx")
     */
    static std::string getFileExtension(const std::string& filePath);

    /**
     * @brief Check OpenCV DNN module availability
     * @return True if DNN module is available
     */
    static bool isDnnAvailable();

    /**
     * @brief Configure detector with optimal parameters based on type
     * @param detector Detector to configure
     * @param type Detector type for parameter selection
     */
    static void configureDetector(IFaceDetector* detector, DetectorType type);

    // Deleted constructors - this is a static utility class
    FaceDetectorFactory() = delete;
    ~FaceDetectorFactory() = delete;
    FaceDetectorFactory(const FaceDetectorFactory&) = delete;
    FaceDetectorFactory& operator=(const FaceDetectorFactory&) = delete;
};

#endif // FACEDETECTORFACTORY_H
