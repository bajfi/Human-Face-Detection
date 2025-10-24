// Copyright (c) 2025 JackLee
//
// This software is released under the MIT License.
// https://opensource.org/licenses/MIT

#include "ConfigurationManager.h"
#include <iostream>
#include <sstream>
#include <fstream>
#include <filesystem>
#include <QStandardPaths>

ConfigurationManager::ConfigurationManager(const std::string& configDirectory)
{
    // Set configuration directory to QStandardPaths default path
    if (configDirectory.empty())
    {
        m_configDirectory =
          QStandardPaths::writableLocation(QStandardPaths::AppConfigLocation)
            .toStdString();
    }
    else
    {
        m_configDirectory = configDirectory;
    }

    m_hardwareInfo = detectHardwareCapabilities();

    // Initialize with default balanced profile
    m_currentProfile =
      createPresetProfile(PerformancePreset::BALANCED, m_hardwareInfo);
    m_currentProfile.name = "Default";
}

bool ConfigurationManager::initialize()
{
    if (m_initialized)
    {
        return true;
    }

    // Ensure config directory exists
    if (!ensureConfigDirectory())
    {
        std::cerr << "Failed to create configuration directory: "
                  << m_configDirectory << std::endl;
        return false;
    }

    // Load existing profiles
    if (!loadAllProfiles())
    {
        std::cerr << "Warning: Failed to load some configuration profiles"
                  << std::endl;
    }

    // Create default profiles if none exist
    if (m_savedProfiles.empty())
    {
        createDefaultProfiles();
    }

    m_initialized = true;
    return true;
}

bool ConfigurationManager::saveDetectionProfile(
  const std::string& name, const DetectionProfile& profile
)
{
    if (name.empty())
    {
        return false;
    }

    DetectionProfile updatedProfile = profile;
    updatedProfile.name = name;
    updatedProfile.modifiedAt = std::chrono::system_clock::now();
    updatedProfile.usageCount++;

    std::filesystem::path filePath = getProfileFilePath(name);
    if (!saveProfileToFile(updatedProfile, filePath))
    {
        return false;
    }

    m_savedProfiles[name] = updatedProfile;
    return true;
}

std::unique_ptr<ConfigurationManager::DetectionProfile> ConfigurationManager::
  loadDetectionProfile(const std::string& name)
{
    if (m_savedProfiles.contains(name))
    {
        return std::make_unique<DetectionProfile>(m_savedProfiles.at(name));
    }

    // Try loading from file
    std::filesystem::path filePath = getProfileFilePath(name);
    return loadProfileFromFile(filePath);
}

bool ConfigurationManager::deleteDetectionProfile(const std::string& name)
{
    // Remove from memory
    m_savedProfiles.erase(name);

    // Remove file with filesystem
    std::filesystem::path filePath = getProfileFilePath(name);
    return std::filesystem::remove(filePath);
}

std::vector<std::string> ConfigurationManager::getAvailableProfiles() const
{
    std::vector<std::string> profiles;
    for (const auto& profile : m_savedProfiles)
    {
        profiles.push_back(profile.first);
    }
    return profiles;
}

ConfigurationManager::DetectionProfile ConfigurationManager::
  createPresetProfile(
    PerformancePreset preset, const HardwareInfo& hardwareInfo
  ) const
{
    DetectionProfile profile;
    profile.name = "Preset";
    profile.description = "Auto-generated preset profile";

    // Configure based on preset type
    switch (preset)
    {
    case PerformancePreset::MAXIMUM_SPEED:
        profile.confidenceThreshold = 0.8f;
        profile.inputResolution = cv::Size(320, 240);
        profile.videoConfig.frameSkipCount = hardwareInfo.hasGpu ? 0 : 2;
        profile.videoConfig.targetResolution = cv::Size(320, 240);
        profile.videoConfig.adaptiveQuality = true;
        break;

    case PerformancePreset::MAXIMUM_QUALITY:
        profile.confidenceThreshold = 0.4f;
        profile.inputResolution = cv::Size(1024, 768);
        profile.videoConfig.frameSkipCount = 0;
        profile.videoConfig.targetResolution = cv::Size(1024, 768);
        profile.videoConfig.adaptiveQuality = false;
        break;

    case PerformancePreset::LOW_MEMORY:
        profile.confidenceThreshold = 0.7f;
        profile.inputResolution = cv::Size(320, 240);
        profile.videoConfig.targetResolution = cv::Size(320, 240);
        profile.memoryConfig.maxPoolSizeMB = 128;
        profile.memoryConfig.initialPoolSizeMB = 64;
        break;

    case PerformancePreset::REAL_TIME:
        profile.confidenceThreshold = 0.6f;
        profile.inputResolution = cv::Size(640, 480);
        profile.videoConfig.frameSkipCount = 0;
        profile.videoConfig.asyncProcessing = true;
        profile.videoConfig.adaptiveQuality = true;
        break;

    case PerformancePreset::BALANCED:
    default:
        profile.confidenceThreshold = 0.6f;
        profile.inputResolution = cv::Size(640, 480);
        profile.videoConfig.frameSkipCount = 0;
        profile.videoConfig.targetResolution = cv::Size(640, 480);
        profile.videoConfig.adaptiveQuality = true;
        break;
    }

    // Optimize for hardware
    optimizeProfileForHardware(profile, hardwareInfo);

    return profile;
}

bool ConfigurationManager::exportSettings(
  const std::string& filePath, bool includeProfiles
) const
{
    // Simple JSON-like export (basic implementation)
    std::ofstream file(filePath);
    if (!file.is_open())
    {
        return false;
    }

    file << "{\n";
    file << "  \"currentProfile\": \"" << m_currentProfile.name << "\",\n";
    file << "  \"hardwareInfo\": {\n";
    file << "    \"hasGpu\": " << (m_hardwareInfo.hasGpu ? "true" : "false")
         << ",\n";
    file << "    \"gpuMemoryMB\": " << m_hardwareInfo.gpuMemoryMB << "\n";
    file << "  }";

    if (includeProfiles && !m_savedProfiles.empty())
    {
        file << ",\n  \"profiles\": [\n";
        bool first = true;
        for (const auto& profile : m_savedProfiles)
        {
            if (!first)
                file << ",\n";
            file << "    {\n";
            file << "      \"name\": \"" << profile.second.name << "\",\n";
            file << "      \"description\": \"" << profile.second.description
                 << "\"\n";
            file << "    }";
            first = false;
        }
        file << "\n  ]";
    }

    file << "\n}\n";
    return true;
}

bool ConfigurationManager::importSettings(
  const std::string& filePath, bool mergeWithExisting
)
{
    // Basic implementation - in a full system this would parse JSON
    std::ifstream file(filePath);
    if (!file.is_open())
    {
        return false;
    }

    if (!mergeWithExisting)
    {
        m_savedProfiles.clear();
    }

    // For now, just return success - full JSON parsing would be implemented
    // here
    return true;
}

const ConfigurationManager::DetectionProfile& ConfigurationManager::
  getCurrentProfile() const
{
    return m_currentProfile;
}

bool ConfigurationManager::setCurrentProfile(const DetectionProfile& profile)
{
    auto errors = validateProfile(profile);
    if (!errors.empty())
    {
        return false;
    }

    m_currentProfile = profile;
    notifyConfigChange(m_currentProfile);

    if (m_autoSaveEnabled)
    {
        saveDetectionProfile(profile.name, profile);
    }

    return true;
}

bool ConfigurationManager::setCurrentProfile(const std::string& name)
{
    auto profile = loadDetectionProfile(name);
    if (!profile)
    {
        return false;
    }

    return setCurrentProfile(*profile);
}

std::vector<std::string> ConfigurationManager::validateProfile(
  const DetectionProfile& profile
) const
{
    std::vector<std::string> errors;

    if (profile.name.empty())
    {
        errors.emplace_back("Profile name cannot be empty");
    }

    if (profile.confidenceThreshold < 0.0f ||
        profile.confidenceThreshold > 1.0f)
    {
        errors.emplace_back("Confidence threshold must be between 0.0 and 1.0");
    }

    if (profile.inputResolution.width < 64 ||
        profile.inputResolution.height < 64)
    {
        errors.emplace_back("Input resolution must be at least 64x64");
    }

    if (profile.videoConfig.frameSkipCount < 0)
    {
        errors.emplace_back("Frame skip count cannot be negative");
    }

    return errors;
}

ConfigurationManager::DetectionProfile ConfigurationManager::
  getHardwareOptimizedProfile() const
{
    PerformancePreset preset = m_hardwareInfo.hasGpu
                                 ? PerformancePreset::REAL_TIME
                                 : PerformancePreset::BALANCED;

    return createPresetProfile(preset, m_hardwareInfo);
}

ConfigurationManager::HardwareInfo ConfigurationManager::
  detectHardwareCapabilities()
{
    HardwareInfo info;

#ifdef OPENCV_DNN_CUDA
    info.hasOpenCVCuda = true;
    info.hasGpu = cv::cuda::getCudaEnabledDeviceCount() > 0;

    if (info.hasGpu)
    {
        cv::cuda::DeviceInfo deviceInfo;
        if (deviceInfo.isCompatible())
        {
            info.gpuName = deviceInfo.name();
            info.gpuMemoryMB = deviceInfo.totalGlobalMem() / (1024 * 1024);
        }
    }
#else
    info.hasOpenCVCuda = false;
    info.hasGpu = false;
#endif

    // Check for DNN module
    try
    {
        cv::dnn::Net net;
        info.hasDnnModule = true;
    }
    catch (...)
    {
        info.hasDnnModule = false;
    }

    // Basic system info (simplified)
    info.systemMemoryMB = 8192; // Default assumption
    info.cpuCores = std::thread::hardware_concurrency();

    return info;
}

void ConfigurationManager::registerConfigChangeCallback(
  ConfigChangeCallback callback
)
{
    m_configChangeCallback = callback;
}

bool ConfigurationManager::updateParameter(
  const std::string& parameterName, const std::string& value
)
{
    auto path = parseParameterPath(parameterName);
    if (path.empty())
    {
        return false;
    }

    bool success = setParameterValue(m_currentProfile, path, value);
    if (success)
    {
        notifyConfigChange(m_currentProfile);
    }

    return success;
}

std::string ConfigurationManager::getParameter(
  const std::string& parameterName
) const
{
    auto path = parseParameterPath(parameterName);
    if (path.empty())
    {
        return "";
    }

    return getParameterValue(m_currentProfile, path);
}

void ConfigurationManager::resetToDefaults(PerformancePreset preset)
{
    m_currentProfile = createPresetProfile(preset, m_hardwareInfo);
    m_currentProfile.name = "Default";
    notifyConfigChange(m_currentProfile);
}

std::unordered_map<std::string, std::string> ConfigurationManager::
  getConfigurationStats() const
{
    std::unordered_map<std::string, std::string> stats;

    stats["profileCount"] = std::to_string(m_savedProfiles.size());
    stats["currentProfile"] = m_currentProfile.name;
    stats["hasGpu"] = m_hardwareInfo.hasGpu ? "true" : "false";
    stats["hasCuda"] = m_hardwareInfo.hasOpenCVCuda ? "true" : "false";
    stats["autoSave"] = m_autoSaveEnabled ? "enabled" : "disabled";

    return stats;
}

void ConfigurationManager::enableAutoSave(bool enable)
{
    m_autoSaveEnabled = enable;
}

ConfigurationManager::DetectionProfile ConfigurationManager::
  getRecommendedProfile(const std::string& usagePattern) const
{
    // Simple pattern matching - in a full system this could use ML
    if (usagePattern.find("speed") != std::string::npos ||
        usagePattern.find("fast") != std::string::npos)
    {
        return createPresetProfile(
          PerformancePreset::MAXIMUM_SPEED, m_hardwareInfo
        );
    }
    if (usagePattern.find("quality") != std::string::npos ||
        usagePattern.find("accuracy") != std::string::npos)
    {
        return createPresetProfile(
          PerformancePreset::MAXIMUM_QUALITY, m_hardwareInfo
        );
    }
    if (usagePattern.find("memory") != std::string::npos ||
        usagePattern.find("ram") != std::string::npos)
    {
        return createPresetProfile(
          PerformancePreset::LOW_MEMORY, m_hardwareInfo
        );
    }
    if (usagePattern.find("realtime") != std::string::npos ||
        usagePattern.find("live") != std::string::npos)
    {
        return createPresetProfile(
          PerformancePreset::REAL_TIME, m_hardwareInfo
        );
    }

    return createPresetProfile(PerformancePreset::BALANCED, m_hardwareInfo);
}

ConfigurationManager::DetectionProfile ConfigurationManager::
  createProfileFromCurrentSettings(const std::string& name) const
{
    DetectionProfile profile = m_currentProfile;
    profile.name = name;
    profile.createdAt = std::chrono::system_clock::now();
    profile.modifiedAt = profile.createdAt;
    profile.usageCount = 0;

    return profile;
}

// Private methods implementation

bool ConfigurationManager::loadAllProfiles()
{
    // Basic implementation - would scan config directory for profile files
    return true;
}

bool ConfigurationManager::saveProfileToFile(
  const DetectionProfile& profile, const std::filesystem::path& filePath
) const
{
    std::ofstream file(filePath);
    if (!file.is_open())
    {
        return false;
    }

    // Simple serialization - in production this would be JSON
    file << "name=" << profile.name << "\n";
    file << "description=" << profile.description << "\n";
    file << "confidenceThreshold=" << profile.confidenceThreshold << "\n";
    file << "inputWidth=" << profile.inputResolution.width << "\n";
    file << "inputHeight=" << profile.inputResolution.height << "\n";

    return true;
}

std::unique_ptr<ConfigurationManager::DetectionProfile> ConfigurationManager::
  loadProfileFromFile(const std::filesystem::path& filePath) const
{
    std::ifstream file(filePath);
    if (!file.is_open())
    {
        return nullptr;
    }

    auto profile = std::make_unique<DetectionProfile>();

    // Simple deserialization - in production this would be JSON
    std::string line;
    while (std::getline(file, line))
    {
        size_t pos = line.find('=');
        if (pos != std::string::npos)
        {
            std::string key = line.substr(0, pos);
            std::string value = line.substr(pos + 1);

            if (key == "name")
            {
                profile->name = value;
            }
            else if (key == "description")
            {
                profile->description = value;
            }
            else if (key == "confidenceThreshold")
            {
                profile->confidenceThreshold = std::stof(value);
            }
        }
    }

    return profile;
}

std::filesystem::path ConfigurationManager::getProfileFilePath(
  const std::string& profileName
) const
{
    return m_configDirectory / (profileName + ".cfg");
}

void ConfigurationManager::createDefaultProfiles()
{
    // Create default profiles for each preset
    auto speedProfile =
      createPresetProfile(PerformancePreset::MAXIMUM_SPEED, m_hardwareInfo);
    speedProfile.name = "Speed";
    speedProfile.description = "Optimized for maximum processing speed";
    m_savedProfiles["Speed"] = speedProfile;

    auto qualityProfile =
      createPresetProfile(PerformancePreset::MAXIMUM_QUALITY, m_hardwareInfo);
    qualityProfile.name = "Quality";
    qualityProfile.description = "Optimized for maximum detection accuracy";
    m_savedProfiles["Quality"] = qualityProfile;

    auto balancedProfile =
      createPresetProfile(PerformancePreset::BALANCED, m_hardwareInfo);
    balancedProfile.name = "Balanced";
    balancedProfile.description = "Balanced speed and accuracy";
    m_savedProfiles["Balanced"] = balancedProfile;
}

void ConfigurationManager::optimizeProfileForHardware(
  DetectionProfile& profile, const HardwareInfo& hardwareInfo
) const
{
    // Enable GPU acceleration if available
    profile.enableGpuAcceleration =
      hardwareInfo.hasGpu && hardwareInfo.hasOpenCVCuda;
    profile.videoConfig.gpuAcceleration = profile.enableGpuAcceleration;

    // Adjust memory settings based on available GPU memory
    if (hardwareInfo.hasGpu && hardwareInfo.gpuMemoryMB > 0)
    {
        if (hardwareInfo.gpuMemoryMB >= 4096)
        { // 4GB+
            profile.memoryConfig.maxPoolSizeMB = 1024;
        }
        else if (hardwareInfo.gpuMemoryMB >= 2048)
        { // 2GB+
            profile.memoryConfig.maxPoolSizeMB = 512;
        }
        else
        {                                                 // < 2GB
            profile.memoryConfig.maxPoolSizeMB = 256;
            profile.inputResolution = cv::Size(480, 360); // Reduce resolution
        }
    }

    // Adjust processing based on CPU cores
    if (hardwareInfo.cpuCores >= 8)
    {
        profile.videoConfig.asyncProcessing = true;
        profile.videoConfig.maxQueueSize = 15;
    }
    else if (hardwareInfo.cpuCores >= 4)
    {
        profile.videoConfig.asyncProcessing = true;
        profile.videoConfig.maxQueueSize = 10;
    }
    else
    {
        profile.videoConfig.maxQueueSize = 5;
    }
}

bool ConfigurationManager::ensureConfigDirectory() const
{
    // Basic directory creation - in production would use filesystem library
    if (std::filesystem::exists(m_configDirectory))
        return true;
    return std::filesystem::create_directories(m_configDirectory);
}

void ConfigurationManager::notifyConfigChange(const DetectionProfile& profile)
{
    if (m_configChangeCallback)
    {
        m_configChangeCallback(profile);
    }
}

std::vector<std::string> ConfigurationManager::parseParameterPath(
  const std::string& parameterName
) const
{
    std::vector<std::string> path;
    std::istringstream iss(parameterName);
    std::string segment;

    while (std::getline(iss, segment, '.'))
    {
        if (!segment.empty())
        {
            path.push_back(segment);
        }
    }

    return path;
}

bool ConfigurationManager::setParameterValue(
  DetectionProfile& profile,
  const std::vector<std::string>& path,
  const std::string& value
) const
{
    if (path.empty())
    {
        return false;
    }

    // Simple parameter setting - in production this would be more robust
    if (path[0] == "confidenceThreshold")
    {
        profile.confidenceThreshold = std::stof(value);
        return true;
    }
    else if (path[0] == "videoConfig" && path.size() > 1)
    {
        if (path[1] == "frameSkipCount")
        {
            profile.videoConfig.frameSkipCount = std::stoi(value);
            return true;
        }
    }

    return false;
}

std::string ConfigurationManager::getParameterValue(
  const DetectionProfile& profile, const std::vector<std::string>& path
) const
{
    if (path.empty())
    {
        return "";
    }

    // Simple parameter getting - in production this would be more robust
    if (path[0] == "confidenceThreshold")
    {
        return std::to_string(profile.confidenceThreshold);
    }
    if (path[0] == "videoConfig" && path.size() > 1)
    {
        if (path[1] == "frameSkipCount")
        {
            return std::to_string(profile.videoConfig.frameSkipCount);
        }
    }

    return "";
}
