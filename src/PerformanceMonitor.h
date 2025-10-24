// Copyright (c) 2025 JackLee
//
// This software is released under the MIT License.
// https://opensource.org/licenses/MIT

#ifndef PERFORMANCEMONITOR_H
#define PERFORMANCEMONITOR_H

#include <chrono>
#include <vector>
#include <atomic>
#include <mutex>
#include <memory>
#include <functional>
#include <queue>
#include <opencv2/opencv.hpp>

/**
 * @brief Performance monitoring and adaptive quality control system
 *
 * This class provides comprehensive performance monitoring capabilities for the
 * face detection system, including frame processing time tracking, memory usage
 * monitoring, adaptive resolution scaling, and intelligent frame dropping
 * strategies.
 *
 * Features:
 * - Real-time performance metrics collection and analysis
 * - Adaptive quality control based on performance feedback
 * - GPU memory usage monitoring
 * - Configurable performance thresholds and responses
 * - Statistical analysis with rolling averages and percentiles
 *
 * Target Performance: Maintain 30+ FPS with < 33ms processing latency
 */
class PerformanceMonitor
{
  public:
    /**
     * @brief Performance metrics structure
     */
    struct Metrics
    {
        // Timing metrics
        std::chrono::milliseconds avgFrameTime{0};
        std::chrono::milliseconds minFrameTime{
          std::chrono::milliseconds::max()
        };
        std::chrono::milliseconds maxFrameTime{0};
        std::chrono::milliseconds p95FrameTime{0}; // 95th percentile
        std::chrono::milliseconds p99FrameTime{0}; // 99th percentile

        // Throughput metrics
        double currentFPS = 0.0;
        double avgFPS = 0.0;
        double minFPS = std::numeric_limits<double>::max();
        double maxFPS = 0.0;

        // Frame statistics
        size_t processedFrames = 0;
        size_t skippedFrames = 0;
        size_t droppedFrames = 0;
        size_t failedFrames = 0;

        // Memory metrics
        size_t gpuMemoryUsed = 0;          // in bytes
        size_t gpuMemoryAvailable = 0;     // in bytes
        float gpuMemoryUtilization = 0.0f; // percentage

        // System resource metrics
        float cpuUsage = 0.0f;       // percentage
        float gpuUsage = 0.0f;       // percentage
        float gpuTemperature = 0.0f; // Celsius

        // Detection quality metrics
        size_t totalDetections = 0;
        double avgDetectionsPerFrame = 0.0;
        float avgConfidence = 0.0f;

        // Adaptive quality status
        cv::Size currentResolution{0, 0};
        int currentFrameSkip = 0;
        float currentConfidenceThreshold = 0.0f;

        Metrics() = default;
    };

    /**
     * @brief Performance configuration and thresholds
     */
    struct Config
    {
        // Target performance thresholds
        std::chrono::milliseconds targetFrameTime{33};        // ~30 FPS
        std::chrono::milliseconds maxAcceptableFrameTime{50}; // 20 FPS minimum
        double minTargetFPS = 25.0;
        double maxTargetFPS = 60.0;

        size_t minPerformanceSampleSize = 10; // Minimum samples for stats
        double performanceDegradationThreshold =
          0.1; // 10% degradation triggers adaptation

        // Memory thresholds
        size_t maxGpuMemoryUsage = 500 * 1024 * 1024; // 500MB
        float maxGpuMemoryUtilization = 0.8f;         // 80%

        // Adaptive quality parameters
        bool enableAdaptiveResolution = true;
        bool enableAdaptiveFrameSkip = true;
        bool enableAdaptiveThreshold = true;

        cv::Size minResolution{320, 240};
        cv::Size maxResolution{1920, 1080};
        int maxFrameSkip = 4; // Skip at most 4 frames
        float minConfidenceThreshold = 0.3f;
        float maxConfidenceThreshold = 0.9f;

        // Monitoring configuration
        size_t historySize = 100; // Number of samples to keep
        std::chrono::seconds reportingInterval{
          5
        }; // Performance report interval
        bool enableSystemMonitoring = false; // Monitor CPU/GPU usage

        Config()
        {
            // Default values are set by member initializers above
        }
    };

    /**
     * @brief Quality adaptation recommendation
     */
    struct AdaptationRecommendation
    {
        enum class Action
        {
            NO_CHANGE,
            REDUCE_RESOLUTION,
            INCREASE_RESOLUTION,
            INCREASE_FRAME_SKIP,
            DECREASE_FRAME_SKIP,
            INCREASE_THRESHOLD,
            DECREASE_THRESHOLD,
            ENABLE_GPU_ACCELERATION,
            DISABLE_GPU_ACCELERATION
        };

        Action action = Action::NO_CHANGE;
        cv::Size recommendedResolution{0, 0};
        int recommendedFrameSkip = 0;
        float recommendedThreshold = 0.0f;
        std::string reasoning;
        float confidence = 0.0f; // Confidence in recommendation (0.0-1.0)

        AdaptationRecommendation() = default;
    };

    /**
     * @brief Performance alert callback function type
     */
    using AlertCallback =
      std::function<void(const std::string& message, int severity)>;

  public:
    explicit PerformanceMonitor(const Config& config = Config{});
    ~PerformanceMonitor() = default;

    // Prevent copying due to threading components
    PerformanceMonitor(const PerformanceMonitor&) = delete;
    PerformanceMonitor& operator=(const PerformanceMonitor&) = delete;

    /**
     * @brief Start performance monitoring
     */
    void start();

    /**
     * @brief Stop performance monitoring
     */
    void stop();

    /**
     * @brief Track frame processing time
     * @param processingTime Time taken to process the frame
     */
    void trackFrameProcessingTime(std::chrono::milliseconds processingTime);

    /**
     * @brief Track frame processing result
     * @param processingTime Processing duration
     * @param detectionCount Number of detections found
     * @param avgConfidence Average confidence of detections
     * @param frameSkipped True if frame was skipped
     * @param frameDropped True if frame was dropped
     * @param processingFailed True if processing failed
     */
    void trackFrameResult(
      std::chrono::milliseconds processingTime,
      size_t detectionCount,
      float avgConfidence,
      bool frameSkipped = false,
      bool frameDropped = false,
      bool processingFailed = false
    );

    /**
     * @brief Update GPU memory usage information
     * @param usedBytes GPU memory currently used
     * @param availableBytes GPU memory available
     */
    void updateGpuMemoryUsage(size_t usedBytes, size_t availableBytes);

    /**
     * @brief Update system resource usage
     * @param cpuUsage CPU usage percentage (0.0-100.0)
     * @param gpuUsage GPU usage percentage (0.0-100.0)
     * @param gpuTemperature GPU temperature in Celsius
     */
    void updateSystemUsage(
      float cpuUsage, float gpuUsage, float gpuTemperature
    );

    /**
     * @brief Update current processing configuration
     * @param resolution Current processing resolution
     * @param frameSkip Current frame skip count
     * @param confidenceThreshold Current confidence threshold
     */
    void updateCurrentConfig(
      const cv::Size& resolution, int frameSkip, float confidenceThreshold
    );

    /**
     * @brief Get current performance metrics
     * @return Current performance metrics
     */
    Metrics getCurrentMetrics() const;

    /**
     * @brief Check if frame should be skipped based on performance
     * @return True if frame should be skipped
     */
    bool shouldSkipFrame() const;

    /**
     * @brief Get adaptive input resolution recommendation
     * @return Recommended resolution for optimal performance
     */
    cv::Size getAdaptiveResolution() const;

    /**
     * @brief Get quality adaptation recommendation
     * @return Recommended quality adjustments
     */
    AdaptationRecommendation getAdaptationRecommendation() const;

    /**
     * @brief Apply adaptation recommendation
     * @param recommendation Recommendation to apply
     * @return True if recommendation was successfully applied
     */
    bool applyAdaptation(const AdaptationRecommendation& recommendation);

    /**
     * @brief Check if performance is currently meeting targets
     * @return True if performance targets are being met
     */
    bool isPerformanceAcceptable() const;

    /**
     * @brief Get performance health score (0.0-1.0)
     * @return Performance health score where 1.0 is optimal
     */
    float getPerformanceHealthScore() const;

    /**
     * @brief Generate performance report
     * @return Detailed performance report string
     */
    std::string generatePerformanceReport() const;

    /**
     * @brief Reset all performance statistics
     */
    void resetStatistics();

    /**
     * @brief Set alert callback for performance issues
     * @param callback Callback function for alerts
     */
    void setAlertCallback(AlertCallback callback);

    /**
     * @brief Update performance monitoring configuration
     * @param config New configuration
     */
    void updateConfig(const Config& config);

    /**
     * @brief Get current configuration
     * @return Current monitoring configuration
     */
    Config getConfig() const;

    /**
     * @brief Enable or disable automatic quality adaptation
     * @param enable True to enable adaptive quality
     */
    void enableAdaptiveQuality(bool enable);

    /**
     * @brief Check if adaptive quality is enabled
     * @return True if adaptive quality is enabled
     */
    bool isAdaptiveQualityEnabled() const;

  private:
    Config m_config;
    mutable std::mutex m_metricsMutex;
    Metrics m_metrics;

    // Performance history for statistical analysis
    std::queue<std::chrono::milliseconds> m_frameTimeHistory;
    std::queue<double> m_fpsHistory;

    // Adaptive quality state
    std::atomic<bool> m_adaptiveQualityEnabled{true};
    mutable std::mutex m_adaptationMutex;
    cv::Size m_currentAdaptiveResolution{640, 480};
    int m_currentAdaptiveFrameSkip = 0;
    float m_currentAdaptiveThreshold = 0.6f;

    // Alert system
    AlertCallback m_alertCallback;
    mutable std::mutex m_alertMutex;
    std::chrono::steady_clock::time_point m_lastAlertTime;

    // Timing
    std::chrono::steady_clock::time_point m_startTime;
    std::chrono::steady_clock::time_point m_lastFrameTime;
    std::atomic<bool> m_monitoring{false};

  private:
    /**
     * @brief Update statistical metrics from history
     */
    void updateStatistics();

    /**
     * @brief Calculate percentile from frame time history
     * @param percentile Percentile to calculate (0.0-1.0)
     * @return Percentile value
     */
    std::chrono::milliseconds calculatePercentile(float percentile) const;

    /**
     * @brief Analyze performance trend
     * @return True if performance is degrading
     */
    bool isPerformanceDegrading() const;

    /**
     * @brief Send performance alert
     * @param message Alert message
     * @param severity Alert severity (0=info, 1=warning, 2=error)
     */
    void sendAlert(const std::string& message, int severity);

    /**
     * @brief Calculate optimal resolution based on current performance
     * @return Recommended resolution
     */
    cv::Size calculateOptimalResolution() const;

    /**
     * @brief Calculate optimal frame skip based on performance
     * @return Recommended frame skip count
     */
    int calculateOptimalFrameSkip() const;

    /**
     * @brief Calculate optimal confidence threshold
     * @return Recommended confidence threshold
     */
    float calculateOptimalThreshold() const;

    /**
     * @brief Validate and clamp resolution to acceptable range
     * @param resolution Input resolution
     * @return Clamped resolution within acceptable range
     */
    cv::Size clampResolution(const cv::Size& resolution) const;
};

#endif // PERFORMANCEMONITOR_H
