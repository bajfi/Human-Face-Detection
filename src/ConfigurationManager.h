// Copyright (c) 2025 JackLee
//
// This software is released under the MIT License.
// https://opensource.org/licenses/MIT

#ifndef CONFIGURATIONMANAGER_H
#define CONFIGURATIONMANAGER_H

#include "VideoFaceDetector.h"
#include "PerformanceMonitor.h"
#include "AdaptiveMemoryPool.h"
#include <opencv2/opencv.hpp>

#include <string>
#include <vector>
#include <memory>
#include <unordered_map>
#include <functional>
#include <filesystem>

/**
 * @brief Configuration management system for face detection profiles and
 * settings
 *
 * This class provides comprehensive configuration management for the face
 * detection system, including detection profiles, performance presets, settings
 * export/import, and runtime configuration updates.
 *
 * Features:
 * - Detection profile management with named presets
 * - Performance optimization presets (Speed, Balanced, Quality)
 * - Configuration export/import in JSON format
 * - Runtime configuration validation and updates
 * - Hardware-specific optimization recommendations
 * - User preference persistence and restoration
 *
 * Configuration Hierarchy:
 * - System defaults (compiled-in)
 * - Hardware-specific optimizations (detected at runtime)
 * - User profiles (saved/loaded)
 * - Session settings (temporary overrides)
 */
class ConfigurationManager
{
  public:
    /**
     * @brief Detection profile containing all face detection settings
     */
    struct DetectionProfile
    {
        std::string name;
        std::string description;

        // Core detection settings
        std::string modelPath;
        std::string detectorType; // "cascade", "dnn", "yunet"

        // Detection parameters
        float confidenceThreshold = 0.6f;
        float nmsThreshold = 0.3f;
        cv::Size minFaceSize{30, 30};
        cv::Size maxFaceSize{300, 300};
        double scaleFactor = 1.1;
        int minNeighbors = 3;

        // Processing settings
        cv::Size inputResolution{640, 480};
        bool enableGpuAcceleration = true;
        bool enablePreprocessing = true;

        // Performance settings
        VideoFaceDetector::ProcessingConfig videoConfig;
        PerformanceMonitor::Config performanceConfig;
        AdaptiveMemoryPool::PoolConfig memoryConfig;

        // Metadata
        std::chrono::system_clock::time_point createdAt;
        std::chrono::system_clock::time_point modifiedAt;
        size_t usageCount = 0;

        DetectionProfile()
          : createdAt(std::chrono::system_clock::now()),
            modifiedAt(std::chrono::system_clock::now())
        {}

        DetectionProfile(const std::string& profileName)
          : name(profileName),
            createdAt(std::chrono::system_clock::now()),
            modifiedAt(std::chrono::system_clock::now())
        {}
    };

    /**
     * @brief Performance preset enumeration
     */
    enum class PerformancePreset
    {
        MAXIMUM_SPEED,   // Prioritize speed over accuracy
        BALANCED,        // Balance between speed and accuracy
        MAXIMUM_QUALITY, // Prioritize accuracy over speed
        LOW_MEMORY,      // Optimize for low memory usage
        HIGH_THROUGHPUT, // Optimize for batch processing
        REAL_TIME,       // Optimize for real-time video processing
        CUSTOM           // User-defined settings
    };

    /**
     * @brief Hardware capability information
     */
    struct HardwareInfo
    {
        bool hasGpu = false;
        size_t gpuMemoryMB = 0;
        std::string gpuName;
        int cudaCores = 0;
        float gpuComputeCapability = 0.0f;

        size_t systemMemoryMB = 0;
        int cpuCores = 0;
        std::string cpuName;

        // OpenCV capabilities
        bool hasOpenCVCuda = false;
        bool hasDnnModule = false;
        std::vector<std::string> availableBackends;

        HardwareInfo()
        {
            // Default values are set by member initializers above
        }
    };

    /**
     * @brief Configuration change callback function type
     */
    using ConfigChangeCallback =
      std::function<void(const DetectionProfile& newProfile)>;

  public:
    explicit ConfigurationManager(const std::string& configDirectory = "");
    ~ConfigurationManager() = default;

    /**
     * @brief Initialize configuration manager
     * @return True if initialization successful
     */
    bool initialize();

    /**
     * @brief Save detection profile with specified name
     * @param name Profile name
     * @param profile Profile configuration
     * @return True if save successful
     */
    bool saveDetectionProfile(
      const std::string& name, const DetectionProfile& profile
    );

    /**
     * @brief Load detection profile by name
     * @param name Profile name
     * @return Loaded profile, or nullptr if not found
     */
    std::unique_ptr<DetectionProfile> loadDetectionProfile(
      const std::string& name
    );

    /**
     * @brief Delete detection profile
     * @param name Profile name to delete
     * @return True if deletion successful
     */
    bool deleteDetectionProfile(const std::string& name);

    /**
     * @brief Get list of available profile names
     * @return Vector of profile names
     */
    std::vector<std::string> getAvailableProfiles() const;

    /**
     * @brief Create profile from performance preset
     * @param preset Performance preset type
     * @param hardwareInfo Hardware information for optimization
     * @return Generated profile
     */
    DetectionProfile createPresetProfile(
      PerformancePreset preset,
      const HardwareInfo& hardwareInfo = HardwareInfo{}
    ) const;

    /**
     * @brief Export settings to file
     * @param filePath Export file path (JSON format)
     * @param includeProfiles True to include all profiles, false for current
     * settings only
     * @return True if export successful
     */
    bool exportSettings(
      const std::string& filePath, bool includeProfiles = true
    ) const;

    /**
     * @brief Import settings from file
     * @param filePath Import file path (JSON format)
     * @param mergeWithExisting True to merge with existing profiles, false to
     * replace
     * @return True if import successful
     */
    bool importSettings(
      const std::string& filePath, bool mergeWithExisting = true
    );

    /**
     * @brief Get current active profile
     * @return Current active profile
     */
    const DetectionProfile& getCurrentProfile() const;

    /**
     * @brief Set active profile
     * @param profile New active profile
     * @return True if profile set successfully
     */
    bool setCurrentProfile(const DetectionProfile& profile);

    /**
     * @brief Set active profile by name
     * @param name Profile name
     * @return True if profile found and set successfully
     */
    bool setCurrentProfile(const std::string& name);

    /**
     * @brief Validate profile configuration
     * @param profile Profile to validate
     * @return Vector of validation error messages (empty if valid)
     */
    std::vector<std::string> validateProfile(
      const DetectionProfile& profile
    ) const;

    /**
     * @brief Get hardware-optimized recommendations for current system
     * @return Recommended profile for current hardware
     */
    DetectionProfile getHardwareOptimizedProfile() const;

    /**
     * @brief Detect current hardware capabilities
     * @return Hardware information structure
     */
    static HardwareInfo detectHardwareCapabilities();

    /**
     * @brief Register callback for configuration changes
     * @param callback Callback function
     */
    void registerConfigChangeCallback(ConfigChangeCallback callback);

    /**
     * @brief Update profile parameter at runtime
     * @param parameterName Parameter name (dot-notation supported)
     * @param value New parameter value
     * @return True if update successful
     */
    bool updateParameter(
      const std::string& parameterName, const std::string& value
    );

    /**
     * @brief Get profile parameter value
     * @param parameterName Parameter name (dot-notation supported)
     * @return Parameter value as string, empty if not found
     */
    std::string getParameter(const std::string& parameterName) const;

    /**
     * @brief Reset profile to system defaults
     * @param preset Preset to reset to (default: BALANCED)
     */
    void resetToDefaults(
      PerformancePreset preset = PerformancePreset::BALANCED
    );

    /**
     * @brief Get configuration statistics
     * @return Configuration usage and performance statistics
     */
    std::unordered_map<std::string, std::string> getConfigurationStats() const;

    /**
     * @brief Enable or disable auto-save of configuration changes
     * @param enable True to enable auto-save
     */
    void enableAutoSave(bool enable);

    /**
     * @brief Get profile recommendation based on usage patterns
     * @param usagePattern Description of intended usage
     * @return Recommended profile
     */
    DetectionProfile getRecommendedProfile(
      const std::string& usagePattern
    ) const;

    /**
     * @brief Create profile from current runtime settings
     * @param name Name for the new profile
     * @return Created profile
     */
    DetectionProfile createProfileFromCurrentSettings(
      const std::string& name
    ) const;

  private:
    std::filesystem::path m_configDirectory;
    DetectionProfile m_currentProfile;
    std::unordered_map<std::string, DetectionProfile> m_savedProfiles;
    HardwareInfo m_hardwareInfo;

    ConfigChangeCallback m_configChangeCallback;
    bool m_autoSaveEnabled = true;
    bool m_initialized = false;

  private:
    /**
     * @brief Load all profiles from configuration directory
     * @return True if loading successful
     */
    bool loadAllProfiles();

    /**
     * @brief Save profile to file
     * @param profile Profile to save
     * @param filePath File path
     * @return True if save successful
     */
    bool saveProfileToFile(
      const DetectionProfile& profile, const std::filesystem::path& filePath
    ) const;

    /**
     * @brief Load profile from file
     * @param filePath File path
     * @return Loaded profile, or nullptr if failed
     */
    std::unique_ptr<DetectionProfile> loadProfileFromFile(
      const std::filesystem::path& filePath
    ) const;

    /**
     * @brief Convert profile to JSON string
     * @param profile Profile to convert
     * @return JSON string representation
     */
    std::string profileToJson(const DetectionProfile& profile) const;

    /**
     * @brief Create profile from JSON string
     * @param json JSON string
     * @return Created profile, or nullptr if parsing failed
     */
    std::unique_ptr<DetectionProfile> profileFromJson(
      const std::string& json
    ) const;

    /**
     * @brief Get profile file path
     * @param profileName Profile name
     * @return Full file path
     */
    std::filesystem::path getProfileFilePath(
      const std::string& profileName
    ) const;

    /**
     * @brief Create system default profiles
     */
    void createDefaultProfiles();

    /**
     * @brief Optimize profile for specific hardware
     * @param profile Profile to optimize
     * @param hardwareInfo Hardware information
     */
    void optimizeProfileForHardware(
      DetectionProfile& profile, const HardwareInfo& hardwareInfo
    ) const;

    /**
     * @brief Validate configuration directory and create if needed
     * @return True if directory is ready
     */
    bool ensureConfigDirectory() const;

    /**
     * @brief Notify configuration change
     * @param profile New profile
     */
    void notifyConfigChange(const DetectionProfile& profile);

    /**
     * @brief Parse dot-notation parameter path
     * @param parameterName Parameter path (e.g., "videoConfig.frameSkipCount")
     * @return Vector of path components
     */
    std::vector<std::string> parseParameterPath(
      const std::string& parameterName
    ) const;

    /**
     * @brief Set parameter value by path
     * @param profile Profile to modify
     * @param path Parameter path components
     * @param value New value
     * @return True if parameter was set successfully
     */
    bool setParameterValue(
      DetectionProfile& profile,
      const std::vector<std::string>& path,
      const std::string& value
    ) const;

    /**
     * @brief Get parameter value by path
     * @param profile Profile to read from
     * @param path Parameter path components
     * @return Parameter value as string
     */
    std::string getParameterValue(
      const DetectionProfile& profile, const std::vector<std::string>& path
    ) const;
};

#endif // CONFIGURATIONMANAGER_H
