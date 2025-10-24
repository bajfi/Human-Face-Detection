// Copyright (c) 2025 JackLee
//
// This software is released under the MIT License.
// https://opensource.org/licenses/MIT

#include "ModelValidator.h"
#include <filesystem>
#include <opencv2/opencv.hpp>

ValidationResult ModelValidator::validateModel(const std::string& modelPath)
{
    // First check if file exists and is accessible
    if (!isFileAccessible(modelPath))
    {
        return ValidationResult(
          false, "Model file not found or not accessible: " + modelPath
        );
    }

    std::filesystem::path fileInfo(modelPath);
    std::string extension = fileInfo.extension().string();

    // Validate based on file extension
    if (extension == ".xml" || extension == ".cascade")
    {
        return validateCascadeFile(modelPath);
    }
    if (extension == ".onnx")
    {
        return validateOnnxFile(modelPath);
    }
    if (extension == ".pb" || extension == ".caffemodel" ||
        extension == ".t7" || extension == ".net" || extension == ".weights")
    {
        // For other DNN formats, perform basic validation
        ValidationResult result(true);
        result.warningMessage =
          "Model format detected but full validation not implemented for ." +
          extension + " files. Please ensure model is compatible.";
        return result;
    }

    return ValidationResult(false, "Unsupported model format: ." + extension);
}

bool ModelValidator::isFileAccessible(const std::string& filePath)
{
    return std::filesystem::exists(filePath) &&
           std::filesystem::is_regular_file(filePath) &&
           std::filesystem::file_size(filePath) > 0;
}

std::string ModelValidator::getDefaultModelPath(bool preferDnn)
{
    // Get current working directory
    std::filesystem::path currentDir = std::filesystem::current_path();

    // Search paths: current directory, parent directory, and some common
    // locations
    std::vector<std::filesystem::path> basePaths = {
      currentDir,               // Current working directory
      currentDir.parent_path(), // Parent directory (for build/executable case)
      std::filesystem::path("."),  // Relative current
      std::filesystem::path(".."), // Relative parent
    };

    // Model subdirectories to search
    std::vector<std::string> modelSubdirs = {
      "models", "data", "assets", "resources"
    };

    std::vector<std::string> dnnModels = {"face_detection_yunet_2023mar.onnx"};
    std::vector<std::string> cascadeModels = {
      "haarcascade_frontalface_alt.xml",
      "haarcascade_frontalface_default.xml",
      "haarcascade_frontalface.xml"
    };

    // Search for models in order of preference
    std::vector<std::vector<std::string>> modelsToSearch =
      preferDnn
        ? std::vector<std::vector<std::string>>{dnnModels, cascadeModels}
        : std::vector<std::vector<std::string>>{cascadeModels, dnnModels};

    // Search in each base path
    for (const auto& basePath : basePaths)
    {
        try
        {
            if (!std::filesystem::exists(basePath))
                continue;

            // Search in model subdirectories
            for (const auto& modelSubdir : modelSubdirs)
            {
                std::filesystem::path dataDir = basePath / modelSubdir;
                if (std::filesystem::exists(dataDir) &&
                    std::filesystem::is_directory(dataDir))
                {
                    for (const auto& modelList : modelsToSearch)
                    {
                        for (const auto& modelName : modelList)
                        {
                            std::filesystem::path modelPath =
                              dataDir / modelName;
                            if (isFileAccessible(modelPath.string()))
                            {
                                return std::filesystem::canonical(modelPath)
                                  .string();
                            }
                        }
                    }
                }
            }

            // Also search directly in the base path
            for (const auto& modelList : modelsToSearch)
            {
                for (const auto& modelName : modelList)
                {
                    std::filesystem::path modelPath = basePath / modelName;
                    if (isFileAccessible(modelPath.string()))
                    {
                        return std::filesystem::canonical(modelPath).string();
                    }
                }
            }
        }
        catch (const std::filesystem::filesystem_error& ex)
        {
            // Continue with next path if this one fails
            continue;
        }
    }

    return ""; // No default model found
}

ValidationResult ModelValidator::validateCascadeFile(const std::string& xmlPath)
{
    ValidationResult result(true);

    try
    {
        // Try to load the cascade file to verify it's valid
        cv::CascadeClassifier testClassifier;
        if (!testClassifier.load(xmlPath))
        {
            return ValidationResult(
              false, "Failed to load cascade file: " + xmlPath
            );
        }

        if (testClassifier.empty())
        {
            return ValidationResult(
              false, "Cascade file is empty or invalid: " + xmlPath
            );
        }

        result.warningMessage = "Cascade file loaded successfully";
    }
    catch (const std::exception& e)
    {
        return ValidationResult(
          false,
          std::format("Exception while validating cascade file: {}", e.what())
        );
    }

    return result;
}

ValidationResult ModelValidator::validateOnnxFile(const std::string& onnxPath)
{
    ValidationResult result(true);

    try
    {
        // Basic file size check - ONNX models shouldn't be too small
        std::filesystem::path filePath(onnxPath);
        auto fileSize = std::filesystem::file_size(filePath);

        if (fileSize < 1024) // Less than 1KB is suspicious
        {
            return ValidationResult(
              false,
              "ONNX file appears to be too small: " + std::to_string(fileSize) +
                " bytes"
            );
        }

        // Try to create a DNN network to validate the ONNX file
        try
        {
            cv::dnn::Net testNet = cv::dnn::readNetFromONNX(onnxPath);
            if (testNet.empty())
            {
                return ValidationResult(
                  false,
                  "Failed to load ONNX model or model is empty: " + onnxPath
                );
            }

            result.warningMessage = "ONNX model loaded successfully";
        }
        catch (const cv::Exception& cvEx)
        {
            return ValidationResult(
              false,
              "OpenCV DNN error loading ONNX model: " + std::string(cvEx.what())
            );
        }
    }
    catch (const std::filesystem::filesystem_error& fsEx)
    {
        return ValidationResult(
          false,
          "Filesystem error accessing ONNX file: " + std::string(fsEx.what())
        );
    }
    catch (const std::exception& e)
    {
        return ValidationResult(
          false,
          "Exception while validating ONNX file: " + std::string(e.what())
        );
    }

    return result;
}
