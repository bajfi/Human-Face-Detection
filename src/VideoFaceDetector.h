// Copyright (c) 2025 JackLee
//
// This software is released under the MIT License.
// https://opensource.org/licenses/MIT

#ifndef VIDEOFACEDETECTOR_H
#define VIDEOFACEDETECTOR_H

#include "IFaceDetector.h"
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#ifdef OPENCV_DNN_CUDA
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudaarithm.hpp>
#endif

#include <memory>
#include <chrono>
#include <atomic>
#include <thread>
#include <queue>
#include <mutex>
#include <condition_variable>
#include <functional>

/**
 * @brief GPU-accelerated video stream face detection with performance
 * optimizations
 *
 * This class extends the basic face detection capabilities to handle video
 * streams efficiently using GPU acceleration, triple buffering, asynchronous
 * processing, and intelligent frame management to achieve real-time
 * performance.
 *
 * Features:
 * - Triple buffering for smooth video processing
 * - Asynchronous GPU processing with overlapping operations
 * - Configurable frame skipping for performance tuning
 * - Pre-allocated GPU memory pools to minimize allocation overhead
 * - Performance monitoring and adaptive quality control
 *
 * Target Performance: 30+ FPS on RTX 3060, < 500MB GPU VRAM for 1080p
 */
class VideoFaceDetector
{
  public:
    /**
     * @brief GPU memory pool for efficient video stream processing
     */
    struct VideoProcessingPool
    {
        static constexpr size_t BUFFER_COUNT = 3;    // Triple buffering

#ifdef OPENCV_DNN_CUDA
        cv::cuda::GpuMat frameBuffers[BUFFER_COUNT]; // Input frame buffers
        cv::cuda::GpuMat preprocessedFrame; // Preprocessed frame buffer
        cv::cuda::GpuMat resizedFrame;      // Resized frame buffer
        cv::cuda::Stream asyncStream;       // CUDA stream for async operations
#endif
        cv::Mat cpuBuffers[BUFFER_COUNT];   // CPU fallback buffers
        std::atomic<size_t> currentBufferIndex{0};    // Current buffer index
        std::atomic<size_t> processingBufferIndex{0}; // Buffer being processed
        cv::Size bufferSize;                          // Allocated buffer size
        bool initialized = false; // Pool initialization status

        VideoProcessingPool() = default;
        ~VideoProcessingPool() = default;

        // Prevent copying due to CUDA resources
        VideoProcessingPool(const VideoProcessingPool&) = delete;
        VideoProcessingPool& operator=(const VideoProcessingPool&) = delete;

        // Enable move semantics
        VideoProcessingPool(VideoProcessingPool&&) = default;
        VideoProcessingPool& operator=(VideoProcessingPool&&) = default;
    };

    /**
     * @brief Video processing configuration
     */
    struct ProcessingConfig
    {
        int frameSkipCount = 0;      // Process every Nth frame (0 = all frames)
        bool asyncProcessing = true; // Enable asynchronous processing
        bool gpuAcceleration = true; // Enable GPU acceleration if available
        cv::Size targetResolution{640, 480}; // Target processing resolution
        int maxQueueSize = 10;               // Maximum frame queue size
        float confidenceThreshold = 0.6f;    // Detection confidence threshold
        bool adaptiveQuality = true;         // Enable adaptive quality control

        ProcessingConfig()
        {
            // Default values are set by member initializers above
        }
    };

    /**
     * @brief Video processing statistics
     */
    struct ProcessingStats
    {
        std::chrono::milliseconds avgProcessingTime{0};
        std::chrono::milliseconds lastFrameTime{0};
        size_t processedFrames = 0;
        size_t skippedFrames = 0;
        size_t droppedFrames = 0;
        double currentFPS = 0.0;
        size_t gpuMemoryUsage = 0; // in bytes
        bool gpuAccelerationActive = false;

        // Performance thresholds
        static constexpr std::chrono::milliseconds TARGET_FRAME_TIME{
          33
        }; // ~30 FPS
        static constexpr size_t MAX_GPU_MEMORY_MB = 500;

        ProcessingStats() = default;
    };

    /**
     * @brief Frame processing result
     */
    struct FrameResult
    {
        cv::Mat processedFrame;
        std::vector<DetectionResult> detections;
        std::chrono::high_resolution_clock::time_point timestamp;
        size_t frameIndex;
        std::chrono::milliseconds processingTime;

        FrameResult()
          : timestamp(std::chrono::high_resolution_clock::now()), frameIndex(0)
        {}
    };

    /**
     * @brief Callback function type for processed frames
     */
    using FrameCallback = std::function<void(const FrameResult&)>;

  public:
    explicit VideoFaceDetector(std::shared_ptr<IFaceDetector> detector);
    ~VideoFaceDetector();

    // Prevent copying due to thread and CUDA resources
    VideoFaceDetector(const VideoFaceDetector&) = delete;
    VideoFaceDetector& operator=(const VideoFaceDetector&) = delete;

    /**
     * @brief Initialize video processing with specified configuration
     * @param config Processing configuration
     * @return True if initialization successful
     */
    bool initialize(const ProcessingConfig& config = ProcessingConfig{});

    /**
     * @brief Start video stream processing
     * @param capture OpenCV VideoCapture object
     * @param callback Callback function for processed frames
     * @return True if processing started successfully
     */
    bool startProcessing(cv::VideoCapture& capture, FrameCallback callback);

    /**
     * @brief Stop video processing
     */
    void stopProcessing();

    /**
     * @brief Process a single video frame
     * @param frame Input frame
     * @return Frame processing result
     */
    FrameResult processSingleFrame(const cv::Mat& frame);

    /**
     * @brief Enable or disable asynchronous processing
     * @param enable True to enable async processing
     */
    void enableAsyncProcessing(bool enable);

    /**
     * @brief Set frame skipping configuration
     * @param skipCount Process every Nth frame (0 = all frames, 1 = every other
     * frame, etc.)
     */
    void setFrameSkipping(int skipCount);

    /**
     * @brief Enable or disable GPU acceleration
     * @param enable True to enable GPU acceleration
     * @return True if GPU acceleration was successfully configured
     */
    bool enableGpuAcceleration(bool enable);

    /**
     * @brief Set target processing resolution
     * @param resolution Target resolution for processing (input will be
     * resized)
     */
    void setTargetResolution(const cv::Size& resolution);

    /**
     * @brief Set confidence threshold for detections
     * @param threshold Minimum confidence score (0.0 to 1.0)
     */
    void setConfidenceThreshold(float threshold);

    /**
     * @brief Enable or disable adaptive quality control
     * @param enable True to enable adaptive quality
     */
    void enableAdaptiveQuality(bool enable);

    /**
     * @brief Get current processing statistics
     * @return Current processing statistics
     */
    ProcessingStats getProcessingStats() const;

    /**
     * @brief Check if video processing is currently active
     * @return True if processing is active
     */
    bool isProcessing() const;

    /**
     * @brief Check if GPU acceleration is available and enabled
     * @return True if GPU acceleration is active
     */
    bool isGpuAccelerationEnabled() const;

    /**
     * @brief Get current processing configuration
     * @return Current configuration
     */
    ProcessingConfig getConfiguration() const;

    /**
     * @brief Update processing configuration during runtime
     * @param config New configuration
     * @return True if configuration was successfully updated
     */
    bool updateConfiguration(const ProcessingConfig& config);

    /**
     * @brief Get recommended configuration for current hardware
     * @param prioritizeSpeed If true, optimize for speed; if false, optimize
     * for accuracy
     * @return Recommended configuration
     */
    static ProcessingConfig getRecommendedConfig(bool prioritizeSpeed = true);

  private:
    std::shared_ptr<IFaceDetector> m_detector;
    ProcessingConfig m_config;
    mutable std::mutex m_configMutex;

    // Video processing components
    std::unique_ptr<VideoProcessingPool> m_memoryPool;
    std::atomic<bool> m_processing{false};
    std::atomic<bool> m_stopRequested{false};

    // Threading components
    std::unique_ptr<std::thread> m_processingThread;
    std::queue<cv::Mat> m_frameQueue;
    std::mutex m_queueMutex;
    std::condition_variable m_queueCondition;

    // Performance tracking
    mutable std::mutex m_statsMutex;
    ProcessingStats m_stats;
    std::chrono::high_resolution_clock::time_point m_lastFrameTime;
    std::atomic<size_t> m_frameCounter{0};

    // Callback for processed frames
    FrameCallback m_frameCallback;

  private:
    /**
     * @brief Initialize GPU memory pool
     * @param frameSize Expected frame size
     * @return True if pool was successfully initialized
     */
    bool initializeMemoryPool(const cv::Size& frameSize);

    /**
     * @brief Main processing thread function
     */
    void processingThreadFunc();

    /**
     * @brief Process frame with GPU acceleration
     * @param frame Input frame
     * @return Processing result
     */
    FrameResult processFrameGpu(const cv::Mat& frame);

    /**
     * @brief Process frame with CPU fallback
     * @param frame Input frame
     * @return Processing result
     */
    FrameResult processFrameCpu(const cv::Mat& frame);

    /**
     * @brief Update processing statistics
     * @param processingTime Time taken for last frame
     */
    void updateStats(std::chrono::milliseconds processingTime);

    /**
     * @brief Check if frame should be skipped based on current configuration
     * @return True if frame should be skipped
     */
    bool shouldSkipFrame() const;

    /**
     * @brief Adapt processing quality based on performance
     */
    void adaptQuality();

    /**
     * @brief Cleanup resources
     */
    void cleanup();

    /**
     * @brief Estimate GPU memory usage
     * @return Estimated memory usage in bytes
     */
    size_t estimateGpuMemoryUsage() const;
};

#endif // VIDEOFACEDETECTOR_H
