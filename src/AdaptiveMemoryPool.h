// Copyright (c) 2025 JackLee
//
// This software is released under the MIT License.
// https://opensource.org/licenses/MIT

#ifndef ADAPTIVEMEMORYPOOL_H
#define ADAPTIVEMEMORYPOOL_H

#include <opencv2/opencv.hpp>
#ifdef OPENCV_DNN_CUDA
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudaarithm.hpp>
#endif

#include <vector>
#include <memory>
#include <mutex>
#include <atomic>
#include <thread>
#include <unordered_map>
#include <chrono>
#include <functional>

/**
 * @brief Adaptive GPU memory pool for efficient video processing
 *
 * This class manages GPU memory allocation for face detection video processing,
 * providing dynamic resizing, memory defragmentation, and optimal pool sizing
 * to minimize allocation overhead and maximize performance.
 *
 * Features:
 * - Dynamic memory pool resizing based on usage patterns
 * - Memory defragmentation to reduce fragmentation overhead
 * - Usage pattern analysis for optimal pool sizing
 * - Automatic fallback to CPU when GPU memory is exhausted
 * - Memory pressure monitoring and adaptive responses
 *
 * Memory Management Strategy:
 * - Pre-allocate common sizes to avoid runtime allocation
 * - Track usage patterns to predict optimal pool configuration
 * - Implement smart defragmentation during idle periods
 * - Provide graceful degradation under memory pressure
 */
class AdaptiveMemoryPool
{
  public:
    /**
     * @brief Memory pool configuration
     */
    struct PoolConfig
    {
        size_t maxPoolSizeMB = 512;     // Maximum pool size in MB
        size_t initialPoolSizeMB = 128; // Initial pool size in MB
        size_t growthStepMB = 64;       // Pool growth step in MB

        // Pool management parameters
        float utilizationThreshold = 0.8f; // Utilization threshold for growth
        float fragmentationThreshold =
          0.3f;                            // Fragmentation threshold for defrag
        size_t minBuffersPerSize = 2;      // Minimum buffers per common size
        size_t maxBuffersPerSize = 8;      // Maximum buffers per common size

        // Monitoring and adaptation
        std::chrono::seconds analysisInterval{
          10
        }; // Usage pattern analysis interval
        std::chrono::seconds defragInterval{30}; // Defragmentation interval
        bool enableAutoResize = true;      // Enable automatic pool resizing
        bool enableDefragmentation = true; // Enable memory defragmentation

        PoolConfig()
        {
            // Default values are set by member initializers above
        }
    };

    /**
     * @brief Memory buffer information
     */
    struct BufferInfo
    {
        cv::Size size;
        int type;   // OpenCV mat type (CV_8UC3, etc.)
        bool isGpu; // True if GPU buffer, false if CPU
        std::chrono::steady_clock::time_point lastUsed;
        size_t usageCount = 0;
        bool inUse = false;

        BufferInfo() : lastUsed(std::chrono::steady_clock::now()) {}

        BufferInfo(const cv::Size& s, int t, bool gpu)
          : size(s),
            type(t),
            isGpu(gpu),
            lastUsed(std::chrono::steady_clock::now())
        {}
    };

    /**
     * @brief Pool statistics
     */
    struct PoolStats
    {
        size_t totalAllocatedBytes = 0;
        size_t totalUsedBytes = 0;
        size_t availableBytes = 0;
        size_t fragmentedBytes = 0;

        size_t totalBuffers = 0;
        size_t usedBuffers = 0;
        size_t availableBuffers = 0;

        float utilizationPercent = 0.0f;
        float fragmentationPercent = 0.0f;

        size_t allocationHits = 0;   // Successful pool allocations
        size_t allocationMisses = 0; // Failed pool allocations (fallback)
        size_t defragmentationCount = 0;
        size_t resizeCount = 0;

        std::chrono::steady_clock::time_point lastDefragmentation;
        std::chrono::steady_clock::time_point lastResize;

        PoolStats()
          : lastDefragmentation(std::chrono::steady_clock::now()),
            lastResize(std::chrono::steady_clock::now())
        {}
    };

    /**
     * @brief Smart buffer handle with automatic return-to-pool
     */
    class BufferHandle
    {
      public:
        BufferHandle() = default;
        BufferHandle(AdaptiveMemoryPool* pool, size_t bufferId);
        ~BufferHandle();

        // Move semantics only (no copying to prevent double-release)
        BufferHandle(const BufferHandle&) = delete;
        BufferHandle& operator=(const BufferHandle&) = delete;
        BufferHandle(BufferHandle&& other) noexcept;
        BufferHandle& operator=(BufferHandle&& other) noexcept;

        // Buffer access
#ifdef OPENCV_DNN_CUDA
        cv::cuda::GpuMat& getGpuMat();
        const cv::cuda::GpuMat& getGpuMat() const;
#endif
        cv::Mat& getCpuMat();
        const cv::Mat& getCpuMat() const;

        bool isValid() const
        {
            return m_pool != nullptr && m_bufferId != SIZE_MAX;
        }

        bool isGpu() const;
        cv::Size getSize() const;
        int getType() const;

        // Transfer operations
        void uploadFrom(const cv::Mat& cpuMat);
        void downloadTo(cv::Mat& cpuMat) const;

      private:
        AdaptiveMemoryPool* m_pool = nullptr;
        size_t m_bufferId = SIZE_MAX;

        void release();
    };

  public:
    explicit AdaptiveMemoryPool(const PoolConfig& config = PoolConfig{});
    ~AdaptiveMemoryPool();

    // Prevent copying due to GPU resources
    AdaptiveMemoryPool(const AdaptiveMemoryPool&) = delete;
    AdaptiveMemoryPool& operator=(const AdaptiveMemoryPool&) = delete;

    /**
     * @brief Initialize the memory pool
     * @return True if initialization successful
     */
    bool initialize();

    /**
     * @brief Shutdown and cleanup the memory pool
     */
    void shutdown();

    /**
     * @brief Acquire a buffer from the pool
     * @param size Required buffer size
     * @param type OpenCV mat type (CV_8UC3, etc.)
     * @param preferGpu True to prefer GPU buffer, false for CPU
     * @return Smart buffer handle
     */
    BufferHandle acquireBuffer(
      const cv::Size& size, int type = CV_8UC3, bool preferGpu = true
    );

    /**
     * @brief Pre-allocate buffers for expected sizes
     * @param expectedSizes Vector of expected buffer sizes
     * @param type OpenCV mat type
     * @param buffersPerSize Number of buffers to allocate per size
     * @return True if pre-allocation successful
     */
    bool preallocateBuffers(
      const std::vector<cv::Size>& expectedSizes,
      int type = CV_8UC3,
      size_t buffersPerSize = 3
    );

    /**
     * @brief Resize the memory pool
     * @param newSizeMB New pool size in MB
     * @return True if resize successful
     */
    bool resizePool(size_t newSizeMB);

    /**
     * @brief Perform memory defragmentation
     * @return True if defragmentation was performed
     */
    bool defragmentMemory();

    /**
     * @brief Get optimal pool size for expected usage patterns
     * @param expectedSizes Expected buffer sizes and their usage frequency
     * @return Recommended pool size in MB
     */
    size_t getOptimalPoolSize(
      const std::vector<std::pair<cv::Size, size_t>>& expectedSizes
    ) const;

    /**
     * @brief Get current pool statistics
     * @return Pool statistics
     */
    PoolStats getStatistics() const;

    /**
     * @brief Check if GPU memory is available
     * @return True if GPU memory is available
     */
    bool isGpuMemoryAvailable() const;

    /**
     * @brief Get available GPU memory
     * @return Available GPU memory in bytes
     */
    size_t getAvailableGpuMemory() const;

    /**
     * @brief Update pool configuration
     * @param config New configuration
     */
    void updateConfig(const PoolConfig& config);

    /**
     * @brief Get current configuration
     * @return Current pool configuration
     */
    PoolConfig getConfig() const;

    /**
     * @brief Enable or disable automatic pool management
     * @param enable True to enable automatic management
     */
    void enableAutoManagement(bool enable);

    /**
     * @brief Force garbage collection of unused buffers
     */
    void garbageCollect();

    /**
     * @brief Analyze usage patterns and optimize pool
     */
    void optimizePool();

    /**
     * @brief Get memory usage report
     * @return Detailed memory usage report
     */
    std::string getMemoryReport() const;

  private:
    PoolConfig m_config;
    mutable std::mutex m_poolMutex;

    // Buffer storage
#ifdef OPENCV_DNN_CUDA
    std::vector<cv::cuda::GpuMat> m_gpuBuffers;
#endif
    std::vector<cv::Mat> m_cpuBuffers;
    std::vector<BufferInfo> m_bufferInfos;
    std::vector<bool> m_bufferAvailable;

    // Pool state
    std::atomic<bool> m_initialized{false};
    std::atomic<bool> m_autoManagementEnabled{true};
    size_t m_currentPoolSizeBytes = 0;
    size_t m_nextBufferId = 0;

    // Statistics and monitoring
    mutable std::mutex m_statsMutex;
    PoolStats m_stats;
    std::unordered_map<std::string, size_t>
      m_usagePatterns; // size->count mapping

    // Background management
    std::unique_ptr<std::thread> m_managementThread;
    std::atomic<bool> m_shutdownRequested{false};

  private:
    /**
     * @brief Find available buffer matching requirements
     * @param size Required size
     * @param type Required type
     * @param preferGpu Prefer GPU buffer
     * @return Buffer ID or SIZE_MAX if not found
     */
    size_t findAvailableBuffer(const cv::Size& size, int type, bool preferGpu);

    /**
     * @brief Allocate new buffer
     * @param size Buffer size
     * @param type OpenCV mat type
     * @param isGpu True for GPU buffer
     * @return Buffer ID or SIZE_MAX if allocation failed
     */
    size_t allocateNewBuffer(const cv::Size& size, int type, bool isGpu);

    /**
     * @brief Release buffer back to pool
     * @param bufferId Buffer ID to release
     */
    void releaseBuffer(size_t bufferId);

    /**
     * @brief Calculate buffer size in bytes
     * @param size Buffer dimensions
     * @param type OpenCV mat type
     * @return Size in bytes
     */
    static size_t calculateBufferSize(const cv::Size& size, int type);

    /**
     * @brief Update usage patterns
     * @param size Buffer size used
     */
    void updateUsagePattern(const cv::Size& size);

    /**
     * @brief Management thread function
     */
    void managementThreadFunc();

    /**
     * @brief Check if defragmentation is needed
     * @return True if defragmentation should be performed
     */
    bool needsDefragmentation() const;

    /**
     * @brief Check if pool resize is needed
     * @return True if pool should be resized
     */
    bool needsResize() const;

    /**
     * @brief Calculate memory fragmentation percentage
     * @return Fragmentation percentage (0.0-1.0)
     */
    float calculateFragmentation() const;

    /**
     * @brief Clean up unused buffers
     * @param maxAge Maximum age for unused buffers
     */
    void cleanupUnusedBuffers(std::chrono::seconds maxAge);

    /**
     * @brief Update pool statistics
     */
    void updateStatistics();

    /**
     * @brief Get size key for usage pattern tracking
     * @param size Buffer size
     * @return String key for size
     */
    static std::string getSizeKey(const cv::Size& size);

    friend class BufferHandle;
};

#endif // ADAPTIVEMEMORYPOOL_H
