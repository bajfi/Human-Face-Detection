// Copyright (c) 2025 JackLee
//
// This software is released under the MIT License.
// https://opensource.org/licenses/MIT

#include "ModelValidator.h"
#include <filesystem>
#include <filesystem>

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
    // Get application directory using std::filesystem
    std::filesystem::path appDir = std::filesystem::current_path();

    // Common search paths relative to application
    std::vector<std::string> searchPaths = {"models"};

    std::vector<std::string> dnnModels = {"face_detection_yunet_2023mar.onnx"};
    std::vector<std::string> cascadeModels = {
      "haarcascade_frontalface_alt.xml", "haarcascade_frontalface_default.xml"
    };

    // Search for models in order of preference
    std::vector<std::vector<std::string>> modelsToSearch =
      preferDnn
        ? std::vector<std::vector<std::string>>{dnnModels, cascadeModels}
        : std::vector<std::vector<std::string>>{cascadeModels, dnnModels};

    for (const auto& searchPath : searchPaths)
    {
        std::filesystem::path dataDir = appDir / searchPath;
        if (std::filesystem::exists(dataDir) &&
            std::filesystem::is_directory(dataDir))
        {
            for (const auto& modelList : modelsToSearch)
            {
                for (const auto& modelName : modelList)
                {
                    std::filesystem::path modelPath = dataDir / modelName;
                    if (isFileAccessible(modelPath.string()))
                    {
                        return modelPath.string();
                    }
                }
            }
        }
    }

    return ""; // No default model found
}

ValidationResult ModelValidator::validateCascadeFile(const std::string& xmlPath)
{
    ValidationResult result(true);

    return result;
}

ValidationResult ModelValidator::validateOnnxFile(const std::string& onnxPath)
{
    ValidationResult result(true);

    return result;
}
