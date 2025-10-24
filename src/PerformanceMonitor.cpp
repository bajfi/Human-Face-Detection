// Copyright (c) 2025 JackLee
//
// This software is released under the MIT License.
// https://opensource.org/licenses/MIT

#include "PerformanceMonitor.h"
#include <algorithm>
#include <numeric>
#include <sstream>
#include <format>
#include <cmath>

#ifdef OPENCV_DNN_CUDA
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudafeatures2d.hpp>
#endif

PerformanceMonitor::PerformanceMonitor(const Config& config)
  : m_config(config),
    m_currentAdaptiveResolution(640, 480),
    m_startTime(std::chrono::steady_clock::now()),
    m_lastFrameTime(std::chrono::steady_clock::now()),
    m_lastAlertTime(std::chrono::steady_clock::now())
{
    // Initialize adaptive resolution to a reasonable default
    m_currentAdaptiveResolution = cv::Size(
      std::clamp(
        640, m_config.minResolution.width, m_config.maxResolution.width
      ),
      std::clamp(
        480, m_config.minResolution.height, m_config.maxResolution.height
      )
    );

    // Initialize metrics
    {
        std::lock_guard<std::mutex> lock(m_metricsMutex);
        m_metrics.currentResolution = m_currentAdaptiveResolution;
        m_metrics.currentFrameSkip = m_currentAdaptiveFrameSkip;
        m_metrics.currentConfidenceThreshold = m_currentAdaptiveThreshold;
    }
}

void PerformanceMonitor::start()
{
    m_monitoring.store(true, std::memory_order_release);
    m_startTime = std::chrono::steady_clock::now();
    m_lastFrameTime = m_startTime;

    resetStatistics();
}

void PerformanceMonitor::stop()
{
    m_monitoring.store(false, std::memory_order_release);
}

void PerformanceMonitor::trackFrameProcessingTime(
  std::chrono::milliseconds processingTime
)
{
    trackFrameResult(processingTime, 0, 0.0f, false, false, false);
}

void PerformanceMonitor::trackFrameResult(
  std::chrono::milliseconds processingTime,
  size_t detectionCount,
  float avgConfidence,
  bool frameSkipped,
  bool frameDropped,
  bool processingFailed
)
{
    if (!m_monitoring.load())
    {
        return;
    }

    auto currentTime = std::chrono::steady_clock::now();
    auto frameInterval = std::chrono::duration_cast<std::chrono::milliseconds>(
      currentTime - m_lastFrameTime
    );

    {
        std::lock_guard<std::mutex> lock(m_metricsMutex);

        // Update frame counts
        if (frameSkipped)
        {
            m_metrics.skippedFrames++;
        }
        else if (frameDropped)
        {
            m_metrics.droppedFrames++;
        }
        else if (processingFailed)
        {
            m_metrics.failedFrames++;
        }
        else
        {
            m_metrics.processedFrames++;

            // Update timing metrics
            m_frameTimeHistory.push(processingTime);
            if (m_frameTimeHistory.size() > m_config.historySize)
            {
                m_frameTimeHistory.pop();
            }

            // Update detection metrics
            m_metrics.totalDetections += detectionCount;
            if (m_metrics.processedFrames > 0)
            {
                m_metrics.avgDetectionsPerFrame =
                  static_cast<double>(m_metrics.totalDetections) /
                  m_metrics.processedFrames;
            }

            // Update confidence metrics (exponential moving average)
            if (detectionCount > 0)
            {
                if (m_metrics.avgConfidence == 0.0f)
                {
                    m_metrics.avgConfidence = avgConfidence;
                }
                else
                {
                    m_metrics.avgConfidence =
                      0.9f * m_metrics.avgConfidence + 0.1f * avgConfidence;
                }
            }
        }

        // Update FPS calculation
        if (frameInterval.count() > 0)
        {
            double instantFPS = 1000.0 / frameInterval.count();
            m_fpsHistory.push(instantFPS);
            if (m_fpsHistory.size() > m_config.historySize)
            {
                m_fpsHistory.pop();
            }

            m_metrics.currentFPS = instantFPS;
        }
    }

    m_lastFrameTime = currentTime;

    // Update statistical metrics
    updateStatistics();

    // Check for performance issues and adapt if needed
    if (m_adaptiveQualityEnabled.load())
    {
        auto recommendation = getAdaptationRecommendation();
        if (recommendation.action !=
            AdaptationRecommendation::Action::NO_CHANGE)
        {
            applyAdaptation(recommendation);
        }
    }

    // Send alerts if performance is problematic
    if (!isPerformanceAcceptable())
    {
        auto now = std::chrono::steady_clock::now();
        auto timeSinceLastAlert =
          std::chrono::duration_cast<std::chrono::seconds>(
            now - m_lastAlertTime
          );

        if (timeSinceLastAlert >= m_config.reportingInterval)
        {
            sendAlert(
              "Performance below acceptable levels: " +
                std::to_string(m_metrics.currentFPS) + " FPS",
              1
            );
            m_lastAlertTime = now;
        }
    }
}

void PerformanceMonitor::updateGpuMemoryUsage(
  size_t usedBytes, size_t availableBytes
)
{
    std::lock_guard<std::mutex> lock(m_metricsMutex);

    m_metrics.gpuMemoryUsed = usedBytes;
    m_metrics.gpuMemoryAvailable = availableBytes;

    if (availableBytes > 0)
    {
        m_metrics.gpuMemoryUtilization =
          static_cast<float>(usedBytes) / availableBytes * 100.0f;
    }
    else
    {
        m_metrics.gpuMemoryUtilization = 0.0f;
    }

    // Send memory usage alert if threshold exceeded
    if (usedBytes > m_config.maxGpuMemoryUsage)
    {
        sendAlert(
          "GPU memory usage exceeded: " +
            std::to_string(usedBytes / (1024 * 1024)) + "MB",
          1
        );
    }
}

void PerformanceMonitor::updateSystemUsage(
  float cpuUsage, float gpuUsage, float gpuTemperature
)
{
    std::lock_guard<std::mutex> lock(m_metricsMutex);

    m_metrics.cpuUsage = cpuUsage;
    m_metrics.gpuUsage = gpuUsage;
    m_metrics.gpuTemperature = gpuTemperature;
}

void PerformanceMonitor::updateCurrentConfig(
  const cv::Size& resolution, int frameSkip, float confidenceThreshold
)
{
    std::lock_guard<std::mutex> lock(m_adaptationMutex);

    m_currentAdaptiveResolution = resolution;
    m_currentAdaptiveFrameSkip = frameSkip;
    m_currentAdaptiveThreshold = confidenceThreshold;

    {
        std::lock_guard<std::mutex> metricsLock(m_metricsMutex);
        m_metrics.currentResolution = resolution;
        m_metrics.currentFrameSkip = frameSkip;
        m_metrics.currentConfidenceThreshold = confidenceThreshold;
    }
}

PerformanceMonitor::Metrics PerformanceMonitor::getCurrentMetrics() const
{
    std::lock_guard<std::mutex> lock(m_metricsMutex);
    return m_metrics;
}

bool PerformanceMonitor::shouldSkipFrame() const
{
    if (!m_adaptiveQualityEnabled.load(std::memory_order_acquire) ||
        !m_config.enableAdaptiveFrameSkip)
    {
        return false;
    }

    // Skip frame if performance is poor and we haven't hit the skip limit
    std::lock_guard<std::mutex> lock(m_metricsMutex);

    bool performancePoor =
      m_metrics.avgFrameTime > m_config.targetFrameTime * 1.5;
    bool canSkipMore = m_currentAdaptiveFrameSkip < m_config.maxFrameSkip;

    return performancePoor && canSkipMore;
}

cv::Size PerformanceMonitor::getAdaptiveResolution() const
{
    if (!m_adaptiveQualityEnabled.load(std::memory_order_acquire) ||
        !m_config.enableAdaptiveResolution)
    {
        return m_currentAdaptiveResolution;
    }

    return calculateOptimalResolution();
}

PerformanceMonitor::AdaptationRecommendation PerformanceMonitor::
  getAdaptationRecommendation() const
{
    AdaptationRecommendation recommendation;

    if (!m_adaptiveQualityEnabled.load(std::memory_order_acquire))
    {
        return recommendation;
    }

    std::lock_guard<std::mutex> lock(m_metricsMutex);

    // Analyze current performance
    bool performancePoor = m_metrics.avgFrameTime > m_config.targetFrameTime;
    bool performanceGood =
      m_metrics.avgFrameTime < m_config.targetFrameTime * 0.7;
    bool memoryPressure = m_metrics.gpuMemoryUtilization >
                          m_config.maxGpuMemoryUtilization * 100.0f;

    recommendation.confidence = 0.7f; // Default confidence

    if (performancePoor)
    {
        // Performance is poor, need to reduce quality
        if (m_config.enableAdaptiveFrameSkip &&
            m_currentAdaptiveFrameSkip < m_config.maxFrameSkip)
        {
            recommendation.action =
              AdaptationRecommendation::Action::INCREASE_FRAME_SKIP;
            recommendation.recommendedFrameSkip =
              std::min(m_currentAdaptiveFrameSkip + 1, m_config.maxFrameSkip);
            recommendation.reasoning =
              "Increasing frame skip to improve performance";
            recommendation.confidence = 0.8f;
        }
        else if (m_config.enableAdaptiveResolution)
        {
            cv::Size newResolution = calculateOptimalResolution();
            if (newResolution.width < m_currentAdaptiveResolution.width)
            {
                recommendation.action =
                  AdaptationRecommendation::Action::REDUCE_RESOLUTION;
                recommendation.recommendedResolution = newResolution;
                recommendation.reasoning =
                  "Reducing resolution to improve performance";
                recommendation.confidence = 0.9f;
            }
        }
        else if (m_config.enableAdaptiveThreshold)
        {
            float newThreshold = std::min(
              m_currentAdaptiveThreshold + 0.1f, m_config.maxConfidenceThreshold
            );
            if (newThreshold > m_currentAdaptiveThreshold)
            {
                recommendation.action =
                  AdaptationRecommendation::Action::INCREASE_THRESHOLD;
                recommendation.recommendedThreshold = newThreshold;
                recommendation.reasoning =
                  "Increasing confidence threshold to reduce processing load";
                recommendation.confidence = 0.6f;
            }
        }
    }
    else if (performanceGood && !memoryPressure)
    {
        // Performance is good, can increase quality
        if (m_config.enableAdaptiveFrameSkip && m_currentAdaptiveFrameSkip > 0)
        {
            recommendation.action =
              AdaptationRecommendation::Action::DECREASE_FRAME_SKIP;
            recommendation.recommendedFrameSkip =
              std::max(m_currentAdaptiveFrameSkip - 1, 0);
            recommendation.reasoning =
              "Decreasing frame skip to improve quality";
            recommendation.confidence = 0.7f;
        }
        else if (m_config.enableAdaptiveResolution)
        {
            cv::Size newResolution = calculateOptimalResolution();
            if (newResolution.width > m_currentAdaptiveResolution.width)
            {
                recommendation.action =
                  AdaptationRecommendation::Action::INCREASE_RESOLUTION;
                recommendation.recommendedResolution = newResolution;
                recommendation.reasoning =
                  "Increasing resolution to improve quality";
                recommendation.confidence = 0.8f;
            }
        }
        else if (m_config.enableAdaptiveThreshold)
        {
            float newThreshold = std::max(
              m_currentAdaptiveThreshold - 0.05f,
              m_config.minConfidenceThreshold
            );
            if (newThreshold < m_currentAdaptiveThreshold)
            {
                recommendation.action =
                  AdaptationRecommendation::Action::DECREASE_THRESHOLD;
                recommendation.recommendedThreshold = newThreshold;
                recommendation.reasoning = "Decreasing confidence threshold to "
                                           "improve detection sensitivity";
                recommendation.confidence = 0.5f;
            }
        }
    }

    if (memoryPressure &&
        recommendation.action == AdaptationRecommendation::Action::NO_CHANGE)
    {
        recommendation.action =
          AdaptationRecommendation::Action::REDUCE_RESOLUTION;
        recommendation.recommendedResolution = calculateOptimalResolution();
        recommendation.reasoning = "Reducing resolution due to memory pressure";
        recommendation.confidence = 0.9f;
    }

    return recommendation;
}

bool PerformanceMonitor::applyAdaptation(
  const AdaptationRecommendation& recommendation
)
{
    if (recommendation.action == AdaptationRecommendation::Action::NO_CHANGE)
    {
        return false;
    }

    std::lock_guard<std::mutex> lock(m_adaptationMutex);

    switch (recommendation.action)
    {
    case AdaptationRecommendation::Action::REDUCE_RESOLUTION:
    case AdaptationRecommendation::Action::INCREASE_RESOLUTION:
        if (recommendation.recommendedResolution.width > 0 &&
            recommendation.recommendedResolution.height > 0)
        {
            m_currentAdaptiveResolution =
              clampResolution(recommendation.recommendedResolution);
            return true;
        }
        break;

    case AdaptationRecommendation::Action::INCREASE_FRAME_SKIP:
    case AdaptationRecommendation::Action::DECREASE_FRAME_SKIP:
        m_currentAdaptiveFrameSkip = std::clamp(
          recommendation.recommendedFrameSkip, 0, m_config.maxFrameSkip
        );
        return true;

    case AdaptationRecommendation::Action::INCREASE_THRESHOLD:
    case AdaptationRecommendation::Action::DECREASE_THRESHOLD:
        m_currentAdaptiveThreshold = std::clamp(
          recommendation.recommendedThreshold,
          m_config.minConfidenceThreshold,
          m_config.maxConfidenceThreshold
        );
        return true;

    default:
        break;
    }

    return false;
}

bool PerformanceMonitor::isPerformanceAcceptable() const
{
    std::lock_guard<std::mutex> lock(m_metricsMutex);

    bool fpsAcceptable = m_metrics.currentFPS >= m_config.minTargetFPS;
    bool frameTimeAcceptable =
      m_metrics.avgFrameTime <= m_config.maxAcceptableFrameTime;
    bool memoryAcceptable =
      m_metrics.gpuMemoryUsed <= m_config.maxGpuMemoryUsage;

    return fpsAcceptable && frameTimeAcceptable && memoryAcceptable;
}

float PerformanceMonitor::getPerformanceHealthScore() const
{
    std::lock_guard<std::mutex> lock(m_metricsMutex);

    float fpsScore =
      std::clamp(m_metrics.currentFPS / m_config.maxTargetFPS, 0.0, 1.0);

    float frameTimeScore = 1.0f;
    if (m_metrics.avgFrameTime > std::chrono::milliseconds(0))
    {
        float targetMs = m_config.targetFrameTime.count();
        float actualMs = m_metrics.avgFrameTime.count();
        frameTimeScore = std::clamp(targetMs / actualMs, 0.0f, 1.0f);
    }

    float memoryScore = 1.0f;
    if (m_config.maxGpuMemoryUsage > 0)
    {
        memoryScore = 1.0f - std::clamp(
                               static_cast<float>(m_metrics.gpuMemoryUsed) /
                                 m_config.maxGpuMemoryUsage,
                               0.0f,
                               1.0f
                             );
    }

    // Weighted average: FPS (40%), Frame Time (40%), Memory (20%)
    return 0.4f * fpsScore + 0.4f * frameTimeScore + 0.2f * memoryScore;
}

std::string PerformanceMonitor::generatePerformanceReport() const
{
    std::lock_guard<std::mutex> lock(m_metricsMutex);

    std::ostringstream report;

    report << "=== Performance Report ===\n";
    report << "Timing Metrics:\n";
    report << std::format("  Current FPS: {:.2f}\n", m_metrics.currentFPS);
    report << std::format("  Average FPS: {:.2f}\n", m_metrics.avgFPS);
    report << std::format(
      "  Frame Time (avg): {}ms\n", m_metrics.avgFrameTime.count()
    );
    report << std::format(
      "  Frame Time (95th): {}ms\n", m_metrics.p95FrameTime.count()
    );
    report << std::format(
      "  Frame Time (99th): {}ms\n", m_metrics.p99FrameTime.count()
    );

    report << "\nFrame Statistics:\n";
    report << std::format("  Processed: {}\n", m_metrics.processedFrames);
    report << std::format("  Skipped: {}\n", m_metrics.skippedFrames);
    report << std::format("  Dropped: {}\n", m_metrics.droppedFrames);
    report << std::format("  Failed: {}\n", m_metrics.failedFrames);

    report << "\nDetection Metrics:\n";
    report << std::format(
      "  Total Detections: {}\n", m_metrics.totalDetections
    );
    report << std::format(
      "  Avg Detections/Frame: {:.2f}\n", m_metrics.avgDetectionsPerFrame
    );
    report << std::format(
      "  Average Confidence: {:.2f}\n", m_metrics.avgConfidence
    );

    report << "\nMemory Usage:\n";
    report << std::format(
      "  GPU Memory Used: {}MB\n", (m_metrics.gpuMemoryUsed / (1024 * 1024))
    );
    report << std::format(
      "  GPU Memory Available: {}MB\n",
      (m_metrics.gpuMemoryAvailable / (1024 * 1024))
    );
    report << std::format(
      "  GPU Memory Utilization: {:.2f}%\n", m_metrics.gpuMemoryUtilization
    );

    report << "\nCurrent Configuration:\n";
    report << std::format(
      "  Resolution: {}x{}\n",
      m_metrics.currentResolution.width,
      m_metrics.currentResolution.height
    );
    report << std::format("  Frame Skip: {}\n", m_metrics.currentFrameSkip);
    report << std::format(
      "  Confidence Threshold: {:.2f}\n", m_metrics.currentConfidenceThreshold
    );

    report << std::format(
      "\nPerformance Health Score: {:.2f}%\n",
      (getPerformanceHealthScore() * 100.0f)
    );
    report << std::format(
      "Performance Acceptable: {}\n", (isPerformanceAcceptable() ? "Yes" : "No")
    );

    return report.str();
}

void PerformanceMonitor::resetStatistics()
{
    std::lock_guard<std::mutex> lock(m_metricsMutex);

    m_metrics = Metrics{};
    m_metrics.currentResolution = m_currentAdaptiveResolution;
    m_metrics.currentFrameSkip = m_currentAdaptiveFrameSkip;
    m_metrics.currentConfidenceThreshold = m_currentAdaptiveThreshold;

    // Clear history
    while (!m_frameTimeHistory.empty())
        m_frameTimeHistory.pop();
    while (!m_fpsHistory.empty())
        m_fpsHistory.pop();
}

void PerformanceMonitor::setAlertCallback(AlertCallback callback)
{
    std::lock_guard<std::mutex> lock(m_alertMutex);
    m_alertCallback = callback;
}

void PerformanceMonitor::updateConfig(const Config& config)
{
    std::lock_guard<std::mutex> lock(m_metricsMutex);
    m_config = config;
}

PerformanceMonitor::Config PerformanceMonitor::getConfig() const
{
    std::lock_guard<std::mutex> lock(m_metricsMutex);
    return m_config;
}

void PerformanceMonitor::enableAdaptiveQuality(bool enable)
{
    m_adaptiveQualityEnabled.store(enable, std::memory_order_release);
}

bool PerformanceMonitor::isAdaptiveQualityEnabled() const
{
    return m_adaptiveQualityEnabled.load(std::memory_order_acquire);
}

void PerformanceMonitor::updateStatistics()
{
    std::lock_guard<std::mutex> lock(m_metricsMutex);

    if (!m_frameTimeHistory.empty())
    {
        // Calculate timing statistics
        std::vector<std::chrono::milliseconds> times;
        std::queue<std::chrono::milliseconds> tempQueue = m_frameTimeHistory;

        while (!tempQueue.empty())
        {
            times.push_back(tempQueue.front());
            tempQueue.pop();
        }

        std::sort(times.begin(), times.end());

        auto sum = std::accumulate(
          times.begin(), times.end(), std::chrono::milliseconds(0)
        );
        m_metrics.avgFrameTime = sum / times.size();
        m_metrics.minFrameTime = times.front();
        m_metrics.maxFrameTime = times.back();
        m_metrics.p95FrameTime = calculatePercentile(0.95f);
        m_metrics.p99FrameTime = calculatePercentile(0.99f);
    }

    if (!m_fpsHistory.empty())
    {
        // Calculate FPS statistics
        std::vector<double> fpsList;
        std::queue<double> tempQueue = m_fpsHistory;

        while (!tempQueue.empty())
        {
            fpsList.push_back(tempQueue.front());
            tempQueue.pop();
        }

        std::sort(fpsList.begin(), fpsList.end());

        double sum = std::accumulate(fpsList.begin(), fpsList.end(), 0.0);
        m_metrics.avgFPS = sum / fpsList.size();
        m_metrics.minFPS = fpsList.front();
        m_metrics.maxFPS = fpsList.back();
    }
}

std::chrono::milliseconds PerformanceMonitor::calculatePercentile(
  float percentile
) const
{
    if (m_frameTimeHistory.empty())
    {
        return std::chrono::milliseconds(0);
    }

    std::vector<std::chrono::milliseconds> times;
    std::queue<std::chrono::milliseconds> tempQueue = m_frameTimeHistory;

    while (!tempQueue.empty())
    {
        times.push_back(tempQueue.front());
        tempQueue.pop();
    }

    std::sort(times.begin(), times.end());

    size_t index = static_cast<size_t>(percentile * (times.size() - 1));
    return times[index];
}

bool PerformanceMonitor::isPerformanceDegrading() const
{
    if (m_fpsHistory.size() < m_config.minPerformanceSampleSize)
    {
        return false; // Not enough data
    }

    // Simple trend analysis: compare recent average with overall average
    std::vector<double> recentFps;
    std::queue<double> tempQueue = m_fpsHistory;

    // Get last performance samples
    size_t skip = m_fpsHistory.size() > m_config.minPerformanceSampleSize
                    ? m_fpsHistory.size() - m_config.minPerformanceSampleSize
                    : 0;
    for (size_t i = 0; i < skip; ++i)
    {
        tempQueue.pop();
    }

    while (!tempQueue.empty())
    {
        recentFps.push_back(tempQueue.front());
        tempQueue.pop();
    }

    double recentAvg =
      std::accumulate(recentFps.begin(), recentFps.end(), 0.0) /
      recentFps.size();

    return recentAvg <
           (m_metrics.avgFPS *
            (1.0 -
             m_config
               .performanceDegradationThreshold)); // 10% degradation threshold
}

void PerformanceMonitor::sendAlert(const std::string& message, int severity)
{
    std::lock_guard<std::mutex> lock(m_alertMutex);

    if (m_alertCallback)
    {
        m_alertCallback(message, severity);
    }
}

cv::Size PerformanceMonitor::calculateOptimalResolution() const
{
    // Simple resolution scaling based on performance
    cv::Size current = m_currentAdaptiveResolution;

    if (m_metrics.avgFrameTime > m_config.targetFrameTime * 1.5)
    {
        // Performance is poor, reduce resolution
        int newWidth = static_cast<int>(current.width * 0.8);
        int newHeight = static_cast<int>(current.height * 0.8);
        return clampResolution(cv::Size(newWidth, newHeight));
    }
    if (m_metrics.avgFrameTime < m_config.targetFrameTime * 0.7)
    {
        // Performance is good, can increase resolution
        int newWidth = static_cast<int>(current.width * 1.2);
        int newHeight = static_cast<int>(current.height * 1.2);
        return clampResolution(cv::Size(newWidth, newHeight));
    }

    return current;
}

int PerformanceMonitor::calculateOptimalFrameSkip() const
{
    if (m_metrics.avgFrameTime > m_config.targetFrameTime * 2)
    {
        return std::min(m_currentAdaptiveFrameSkip + 2, m_config.maxFrameSkip);
    }
    if (m_metrics.avgFrameTime > m_config.targetFrameTime * 1.3)
    {
        return std::min(m_currentAdaptiveFrameSkip + 1, m_config.maxFrameSkip);
    }
    if (m_metrics.avgFrameTime < m_config.targetFrameTime * 0.7)
    {
        return std::max(m_currentAdaptiveFrameSkip - 1, 0);
    }

    return m_currentAdaptiveFrameSkip;
}

float PerformanceMonitor::calculateOptimalThreshold() const
{
    // Adjust threshold based on detection count and performance
    if (m_metrics.avgDetectionsPerFrame > 10.0 &&
        m_metrics.avgFrameTime > m_config.targetFrameTime)
    {
        // Too many detections slowing us down
        return std::min(
          m_currentAdaptiveThreshold + 0.1f, m_config.maxConfidenceThreshold
        );
    }
    if (m_metrics.avgDetectionsPerFrame < 1.0 &&
        m_metrics.avgFrameTime < m_config.targetFrameTime * 0.8)
    {
        // Too few detections and we have performance headroom
        return std::max(
          m_currentAdaptiveThreshold - 0.05f, m_config.minConfidenceThreshold
        );
    }

    return m_currentAdaptiveThreshold;
}

cv::Size PerformanceMonitor::clampResolution(const cv::Size& resolution) const
{
    int width = std::clamp(
      resolution.width,
      m_config.minResolution.width,
      m_config.maxResolution.width
    );
    int height = std::clamp(
      resolution.height,
      m_config.minResolution.height,
      m_config.maxResolution.height
    );

    // Ensure dimensions are even (helps with some processing algorithms)
    width = (width / 2) * 2;
    height = (height / 2) * 2;

    return cv::Size(width, height);
}
