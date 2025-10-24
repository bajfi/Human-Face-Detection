// Copyright (c) 2025 JackLee
//
// This software is released under the MIT License.
// https://opensource.org/licenses/MIT

#ifndef MODELVALIDATOR_H
#define MODELVALIDATOR_H

#include <string>
#include <vector>

/**
 * @brief Validation result structure
 */
struct ValidationResult
{
    bool isValid;
    std::string errorMessage;
    std::string warningMessage;

    ValidationResult(
      bool valid = false,
      const std::string& error = "",
      const std::string& warning = ""
    )
      : isValid(valid), errorMessage(error), warningMessage(warning)
    {}
};

/**
 * @brief Utility class for validating face detection models
 *
 * This class provides comprehensive validation for different model formats,
 * ensuring models are properly structured and compatible with the detection
 * system.
 */
class ModelValidator
{
  public:
    /**
     * @brief Validate a model file
     * @param modelPath Path to the model file
     * @return ValidationResult with validation status and messages
     */
    static ValidationResult validateModel(const std::string& modelPath);

    /**
     * @brief Check if a file exists and is readable
     * @param filePath Path to check
     * @return True if file exists and is readable
     */
    static bool isFileAccessible(const std::string& filePath);

    /**
     * @brief Get recommended model path from data directory
     * @param preferDnn If true, prefer DNN models over cascade
     * @return Path to recommended model or empty string if none found
     */
    static std::string getDefaultModelPath(bool preferDnn = true);

    /**
     * @brief Validate XML cascade file structure
     * @param xmlPath Path to XML file
     * @return ValidationResult with specific cascade validation
     */
    static ValidationResult validateCascadeFile(const std::string& xmlPath);

    /**
     * @brief Validate ONNX model file
     * @param onnxPath Path to ONNX file
     * @return ValidationResult with specific ONNX validation
     */
    static ValidationResult validateOnnxFile(const std::string& onnxPath);

  private:
    ModelValidator() = delete;
    ~ModelValidator() = delete;
};

#endif // MODELVALIDATOR_H
