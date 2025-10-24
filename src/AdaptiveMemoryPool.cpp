// Copyright (c) 2025 JackLee
//
// This software is released under the MIT License.
// https://opensource.org/licenses/MIT

#include "AdaptiveMemoryPool.h"
#include <algorithm>
#include <sstream>
#include <iomanip>
#include <thread>

#ifdef OPENCV_DNN_CUDA
#include <opencv2/cudaimgproc.hpp>
#endif

AdaptiveMemoryPool::AdaptiveMemoryPool(const PoolConfig& config)
  : m_config(config)
{
    m_currentPoolSizeBytes = m_config.initialPoolSizeMB * 1024 * 1024;
}

AdaptiveMemoryPool::~AdaptiveMemoryPool()
{
    shutdown();
}

bool AdaptiveMemoryPool::initialize()
{
    if (m_initialized.load())
    {
        return true;
    }

    std::lock_guard<std::mutex> lock(m_poolMutex);

    try
    {
// Initialize buffer storage
#ifdef OPENCV_DNN_CUDA
        m_gpuBuffers.clear();
#endif
        m_cpuBuffers.clear();
        m_bufferInfos.clear();
        m_bufferAvailable.clear();

        // Reserve initial capacity
        size_t initialCapacity = 50; // Start with reasonable capacity
#ifdef OPENCV_DNN_CUDA
        m_gpuBuffers.reserve(initialCapacity);
#endif
        m_cpuBuffers.reserve(initialCapacity);
        m_bufferInfos.reserve(initialCapacity);
        m_bufferAvailable.reserve(initialCapacity);

        // Reset statistics
        {
            std::lock_guard<std::mutex> statsLock(m_statsMutex);
            m_stats = PoolStats{};
        }

        m_initialized.store(true);

        // Start management thread if auto management is enabled
        if (m_autoManagementEnabled.load())
        {
            m_shutdownRequested.store(false);
            m_managementThread = std::make_unique<std::thread>(
              &AdaptiveMemoryPool::managementThreadFunc, this
            );
        }

        return true;
    }
    catch (const std::exception& e)
    {
        return false;
    }
}

void AdaptiveMemoryPool::shutdown()
{
    if (!m_initialized.load())
    {
        return;
    }

    // Signal shutdown to management thread
    m_shutdownRequested.store(true);

    // Wait for management thread to finish
    if (m_managementThread && m_managementThread->joinable())
    {
        m_managementThread->join();
        m_managementThread.reset();
    }

    std::lock_guard<std::mutex> lock(m_poolMutex);

    // Clear all buffers
#ifdef OPENCV_DNN_CUDA
    m_gpuBuffers.clear();
#endif
    m_cpuBuffers.clear();
    m_bufferInfos.clear();
    m_bufferAvailable.clear();

    m_usagePatterns.clear();
    m_currentPoolSizeBytes = 0;
    m_nextBufferId = 0;

    m_initialized.store(false);
}

AdaptiveMemoryPool::BufferHandle AdaptiveMemoryPool::acquireBuffer(
  const cv::Size& size, int type, bool preferGpu
)
{
    if (!m_initialized.load())
    {
        return BufferHandle{};
    }

    std::lock_guard<std::mutex> lock(m_poolMutex);

    // Try to find existing available buffer
    size_t bufferId = findAvailableBuffer(size, type, preferGpu);

    if (bufferId == SIZE_MAX)
    {
        // No suitable buffer found, try to allocate new one
        bufferId = allocateNewBuffer(size, type, preferGpu);

        if (bufferId == SIZE_MAX)
        {
            // GPU allocation failed, try CPU fallback
            if (preferGpu)
            {
                bufferId = allocateNewBuffer(size, type, false);
            }

            if (bufferId == SIZE_MAX)
            {
                // Complete allocation failure
                std::lock_guard<std::mutex> statsLock(m_statsMutex);
                m_stats.allocationMisses++;
                return BufferHandle{};
            }
        }
    }

    // Mark buffer as in use
    if (bufferId < m_bufferAvailable.size())
    {
        m_bufferAvailable[bufferId] = false;
        m_bufferInfos[bufferId].inUse = true;
        m_bufferInfos[bufferId].lastUsed = std::chrono::steady_clock::now();
        m_bufferInfos[bufferId].usageCount++;
    }

    // Update statistics
    {
        std::lock_guard<std::mutex> statsLock(m_statsMutex);
        m_stats.allocationHits++;
        m_stats.usedBuffers++;
    }

    // Update usage patterns
    updateUsagePattern(size);

    return BufferHandle(this, bufferId);
}

bool AdaptiveMemoryPool::preallocateBuffers(
  const std::vector<cv::Size>& expectedSizes, int type, size_t buffersPerSize
)
{
    if (!m_initialized.load())
    {
        return false;
    }

    std::lock_guard<std::mutex> lock(m_poolMutex);

    bool success = true;

    for (const auto& size : expectedSizes)
    {
        // Calculate how many we need vs how many we have
        size_t existingCount = 0;
        for (const auto& info : m_bufferInfos)
        {
            if (info.size == size && info.type == type)
            {
                existingCount++;
            }
        }

        size_t neededCount = (existingCount < buffersPerSize)
                               ? (buffersPerSize - existingCount)
                               : 0;

        // Allocate GPU buffers first, then CPU fallbacks
        for (size_t i = 0; i < neededCount; ++i)
        {
            size_t bufferId =
              allocateNewBuffer(size, type, true); // Try GPU first
            if (bufferId == SIZE_MAX)
            {
                bufferId =
                  allocateNewBuffer(size, type, false); // Fallback to CPU
                if (bufferId == SIZE_MAX)
                {
                    success = false;
                }
            }
        }
    }

    updateStatistics();
    return success;
}

bool AdaptiveMemoryPool::resizePool(size_t newSizeMB)
{
    if (!m_initialized.load())
    {
        return false;
    }

    std::lock_guard<std::mutex> lock(m_poolMutex);

    size_t newSizeBytes = newSizeMB * 1024 * 1024;
    size_t currentUsedBytes = 0;

    // Calculate current used memory
    for (size_t i = 0; i < m_bufferInfos.size(); ++i)
    {
        if (!m_bufferAvailable[i])
        { // Buffer is in use
            currentUsedBytes +=
              calculateBufferSize(m_bufferInfos[i].size, m_bufferInfos[i].type);
        }
    }

    if (newSizeBytes < currentUsedBytes)
    {
        // Cannot resize to smaller than currently used memory
        return false;
    }

    m_currentPoolSizeBytes = newSizeBytes;
    m_config.maxPoolSizeMB = newSizeMB;

    // If we're shrinking, perform garbage collection
    if (newSizeBytes < m_stats.totalAllocatedBytes)
    {
        garbageCollect();
    }

    {
        std::lock_guard<std::mutex> statsLock(m_statsMutex);
        m_stats.resizeCount++;
        m_stats.lastResize = std::chrono::steady_clock::now();
    }

    updateStatistics();
    return true;
}

bool AdaptiveMemoryPool::defragmentMemory()
{
    if (!m_initialized.load() || !m_config.enableDefragmentation)
    {
        return false;
    }

    std::lock_guard<std::mutex> lock(m_poolMutex);

    // Simple defragmentation: remove unused buffers that haven't been used
    // recently
    auto now = std::chrono::steady_clock::now();
    std::chrono::minutes maxAge{5}; // Remove buffers unused for 5+ minutes

    size_t removedCount = 0;

    for (size_t i = 0; i < m_bufferInfos.size();)
    {
        bool shouldRemove = false;

        if (m_bufferAvailable[i] && !m_bufferInfos[i].inUse)
        {
            auto age = std::chrono::duration_cast<std::chrono::minutes>(
              now - m_bufferInfos[i].lastUsed
            );
            if (age > maxAge && m_bufferInfos[i].usageCount < 3)
            { // Rarely used
                shouldRemove = true;
            }
        }

        if (shouldRemove)
        {
            // Remove buffer at index i
#ifdef OPENCV_DNN_CUDA
            if (i < m_gpuBuffers.size())
            {
                m_gpuBuffers.erase(m_gpuBuffers.begin() + i);
            }
#endif
            if (i < m_cpuBuffers.size())
            {
                m_cpuBuffers.erase(m_cpuBuffers.begin() + i);
            }

            m_bufferInfos.erase(m_bufferInfos.begin() + i);
            m_bufferAvailable.erase(m_bufferAvailable.begin() + i);

            removedCount++;

            // Adjust indices for remaining buffers (simple approach)
            // In a production system, you might use a more sophisticated
            // approach
        }
        else
        {
            ++i;
        }
    }

    if (removedCount > 0)
    {
        std::lock_guard<std::mutex> statsLock(m_statsMutex);
        m_stats.defragmentationCount++;
        m_stats.lastDefragmentation = now;
        updateStatistics();
        return true;
    }

    return false;
}

size_t AdaptiveMemoryPool::getOptimalPoolSize(
  const std::vector<std::pair<cv::Size, size_t>>& expectedSizes
) const
{
    size_t totalBytes = 0;

    for (const auto& sizeFreq : expectedSizes)
    {
        const cv::Size& size = sizeFreq.first;
        size_t frequency = sizeFreq.second;

        size_t bufferSize =
          calculateBufferSize(size, CV_8UC3); // Assume BGR format
        size_t buffersNeeded = std::min(frequency, m_config.maxBuffersPerSize);

        totalBytes += bufferSize * buffersNeeded;
    }

    // Add 25% overhead for growth and fragmentation
    totalBytes = static_cast<size_t>(totalBytes * 1.25);

    // Convert to MB and clamp to configured limits
    size_t sizeMB = totalBytes / (1024 * 1024);
    return std::clamp(
      sizeMB, m_config.initialPoolSizeMB, m_config.maxPoolSizeMB
    );
}

AdaptiveMemoryPool::PoolStats AdaptiveMemoryPool::getStatistics() const
{
    std::lock_guard<std::mutex> lock(m_statsMutex);
    return m_stats;
}

bool AdaptiveMemoryPool::isGpuMemoryAvailable() const
{
#ifdef OPENCV_DNN_CUDA
    return cv::cuda::getCudaEnabledDeviceCount() > 0;
#else
    return false;
#endif
}

size_t AdaptiveMemoryPool::getAvailableGpuMemory() const
{
#ifdef OPENCV_DNN_CUDA
    try
    {
        size_t free, total;
        cv::cuda::DeviceInfo deviceInfo;
        if (deviceInfo.isCompatible())
        {
            // Note: OpenCV doesn't provide direct access to cudaMemGetInfo
            // This is a placeholder - in practice you might use CUDA runtime
            // API
            return m_currentPoolSizeBytes - m_stats.totalUsedBytes;
        }
    }
    catch (...)
    {
        // Ignore errors and return 0
    }
#endif
    return 0;
}

void AdaptiveMemoryPool::updateConfig(const PoolConfig& config)
{
    std::lock_guard<std::mutex> lock(m_poolMutex);
    m_config = config;
}

AdaptiveMemoryPool::PoolConfig AdaptiveMemoryPool::getConfig() const
{
    std::lock_guard<std::mutex> lock(m_poolMutex);
    return m_config;
}

void AdaptiveMemoryPool::enableAutoManagement(bool enable)
{
    m_autoManagementEnabled.store(enable);

    if (enable && m_initialized.load() && !m_managementThread)
    {
        m_shutdownRequested.store(false);
        m_managementThread = std::make_unique<std::thread>(
          &AdaptiveMemoryPool::managementThreadFunc, this
        );
    }
}

void AdaptiveMemoryPool::garbageCollect()
{
    cleanupUnusedBuffers(
      std::chrono::seconds(30)
    ); // Remove buffers unused for 30+ seconds
}

void AdaptiveMemoryPool::optimizePool()
{
    if (!m_initialized.load())
    {
        return;
    }

    // Analyze usage patterns and optimize allocation
    std::vector<std::pair<cv::Size, size_t>> patterns;

    {
        std::lock_guard<std::mutex> lock(m_poolMutex);
        for (const auto& pattern : m_usagePatterns)
        {
            cv::Size size;
            std::istringstream iss(pattern.first);
            std::string token;
            if (std::getline(iss, token, 'x'))
            {
                size.width = std::stoi(token);
                if (std::getline(iss, token))
                {
                    size.height = std::stoi(token);
                    patterns.emplace_back(size, pattern.second);
                }
            }
        }
    }

    if (!patterns.empty())
    {
        size_t optimalSize = getOptimalPoolSize(patterns);
        if (optimalSize != m_config.maxPoolSizeMB)
        {
            resizePool(optimalSize);
        }

        // Pre-allocate commonly used sizes
        std::vector<cv::Size> commonSizes;
        std::sort(
          patterns.begin(),
          patterns.end(),
          [](const auto& a, const auto& b)
          {
              return a.second > b.second;
          }
        );

        for (size_t i = 0; i < std::min(patterns.size(), size_t(5)); ++i)
        {
            commonSizes.push_back(patterns[i].first);
        }

        if (!commonSizes.empty())
        {
            preallocateBuffers(
              commonSizes, CV_8UC3, m_config.minBuffersPerSize
            );
        }
    }
}

std::string AdaptiveMemoryPool::getMemoryReport() const
{
    auto stats = getStatistics();
    std::ostringstream report;

    report << std::fixed << std::setprecision(2);
    report << "=== Adaptive Memory Pool Report ===\n";
    report << "Pool Configuration:\n";
    report << "  Max Pool Size: " << m_config.maxPoolSizeMB << " MB\n";
    report << "  Current Pool Size: "
           << (m_currentPoolSizeBytes / (1024 * 1024)) << " MB\n";

    report << "\nMemory Usage:\n";
    report << "  Total Allocated: "
           << (stats.totalAllocatedBytes / (1024 * 1024)) << " MB\n";
    report << "  Currently Used: " << (stats.totalUsedBytes / (1024 * 1024))
           << " MB\n";
    report << "  Available: " << (stats.availableBytes / (1024 * 1024))
           << " MB\n";
    report << "  Utilization: " << stats.utilizationPercent << "%\n";
    report << "  Fragmentation: " << stats.fragmentationPercent << "%\n";

    report << "\nBuffer Statistics:\n";
    report << "  Total Buffers: " << stats.totalBuffers << "\n";
    report << "  Used Buffers: " << stats.usedBuffers << "\n";
    report << "  Available Buffers: " << stats.availableBuffers << "\n";

    report << "\nPerformance Metrics:\n";
    report << "  Allocation Hits: " << stats.allocationHits << "\n";
    report << "  Allocation Misses: " << stats.allocationMisses << "\n";
    report << "  Hit Rate: "
           << (stats.allocationHits + stats.allocationMisses > 0
                 ? (100.0 * stats.allocationHits) /
                     (stats.allocationHits + stats.allocationMisses)
                 : 0.0)
           << "%\n";
    report << "  Defragmentations: " << stats.defragmentationCount << "\n";
    report << "  Pool Resizes: " << stats.resizeCount << "\n";

    return report.str();
}

size_t AdaptiveMemoryPool::findAvailableBuffer(
  const cv::Size& size, int type, bool preferGpu
)
{
    for (size_t i = 0; i < m_bufferInfos.size(); ++i)
    {
        if (m_bufferAvailable[i] && !m_bufferInfos[i].inUse &&
            m_bufferInfos[i].size == size && m_bufferInfos[i].type == type)
        {
            // Check GPU preference
            if (preferGpu && !m_bufferInfos[i].isGpu)
            {
                continue; // Keep looking for GPU buffer
            }

            return i;
        }
    }

    // If preferGpu and no GPU buffer found, try CPU
    if (preferGpu)
    {
        for (size_t i = 0; i < m_bufferInfos.size(); ++i)
        {
            if (m_bufferAvailable[i] && !m_bufferInfos[i].inUse &&
                m_bufferInfos[i].size == size &&
                m_bufferInfos[i].type == type && !m_bufferInfos[i].isGpu)
            {
                return i;
            }
        }
    }

    return SIZE_MAX; // Not found
}

size_t AdaptiveMemoryPool::allocateNewBuffer(
  const cv::Size& size, int type, bool isGpu
)
{
    try
    {
        size_t bufferSize = calculateBufferSize(size, type);

        // Check if we have space in the pool
        if (m_stats.totalAllocatedBytes + bufferSize > m_currentPoolSizeBytes)
        {
            return SIZE_MAX; // Pool is full
        }

        size_t bufferId = m_nextBufferId++;

        // Extend vectors if needed
        if (bufferId >= m_bufferInfos.size())
        {
#ifdef OPENCV_DNN_CUDA
            m_gpuBuffers.resize(bufferId + 1);
#endif
            m_cpuBuffers.resize(bufferId + 1);
            m_bufferInfos.resize(bufferId + 1);
            m_bufferAvailable.resize(bufferId + 1);
        }

        // Allocate the buffer
        if (isGpu)
        {
#ifdef OPENCV_DNN_CUDA
            if (cv::cuda::getCudaEnabledDeviceCount() > 0)
            {
                m_gpuBuffers[bufferId].create(size, type);
                m_bufferInfos[bufferId] = BufferInfo(size, type, true);
            }
            else
            {
                return SIZE_MAX; // GPU not available
            }
#else
            return SIZE_MAX; // GPU support not compiled
#endif
        }
        else
        {
            m_cpuBuffers[bufferId] = cv::Mat::zeros(size, type);
            m_bufferInfos[bufferId] = BufferInfo(size, type, false);
        }

        m_bufferAvailable[bufferId] = true;

        // Update statistics
        {
            std::lock_guard<std::mutex> statsLock(m_statsMutex);
            m_stats.totalAllocatedBytes += bufferSize;
            m_stats.totalBuffers++;
            m_stats.availableBuffers++;
        }

        return bufferId;
    }
    catch (const cv::Exception& e)
    {
        return SIZE_MAX; // Allocation failed
    }
}

void AdaptiveMemoryPool::releaseBuffer(size_t bufferId)
{
    if (bufferId >= m_bufferAvailable.size())
    {
        return;
    }

    std::lock_guard<std::mutex> lock(m_poolMutex);

    if (!m_bufferAvailable[bufferId])
    {
        m_bufferAvailable[bufferId] = true;
        m_bufferInfos[bufferId].inUse = false;

        std::lock_guard<std::mutex> statsLock(m_statsMutex);
        m_stats.usedBuffers--;
        m_stats.availableBuffers++;
    }
}

size_t AdaptiveMemoryPool::calculateBufferSize(const cv::Size& size, int type)
{
    int depth = CV_MAT_DEPTH(type);
    int channels = CV_MAT_CN(type);

    size_t elementSize = 0;
    switch (depth)
    {
    case CV_8U:
    case CV_8S:
        elementSize = 1;
        break;
    case CV_16U:
    case CV_16S:
        elementSize = 2;
        break;
    case CV_32S:
    case CV_32F:
        elementSize = 4;
        break;
    case CV_64F:
        elementSize = 8;
        break;
    default:
        elementSize = 1;
    }

    return size.width * size.height * channels * elementSize;
}

void AdaptiveMemoryPool::updateUsagePattern(const cv::Size& size)
{
    std::string sizeKey = getSizeKey(size);
    m_usagePatterns[sizeKey]++;
}

void AdaptiveMemoryPool::managementThreadFunc()
{
    while (!m_shutdownRequested.load())
    {
        try
        {
            // Sleep for analysis interval
            std::this_thread::sleep_for(m_config.analysisInterval);

            if (m_shutdownRequested.load())
            {
                break;
            }

            // Perform periodic maintenance
            updateStatistics();

            if (needsDefragmentation())
            {
                defragmentMemory();
            }

            if (needsResize())
            {
                optimizePool();
            }

            // Periodic garbage collection
            garbageCollect();
        }
        catch (const std::exception& e)
        {
            // Log error and continue
            std::this_thread::sleep_for(std::chrono::seconds(1));
        }
    }
}

bool AdaptiveMemoryPool::needsDefragmentation() const
{
    if (!m_config.enableDefragmentation)
    {
        return false;
    }

    std::lock_guard<std::mutex> lock(m_statsMutex);

    auto now = std::chrono::steady_clock::now();
    auto timeSinceLastDefrag = std::chrono::duration_cast<std::chrono::seconds>(
      now - m_stats.lastDefragmentation
    );

    return (timeSinceLastDefrag >= m_config.defragInterval) &&
           (m_stats.fragmentationPercent > m_config.fragmentationThreshold);
}

bool AdaptiveMemoryPool::needsResize() const
{
    if (!m_config.enableAutoResize)
    {
        return false;
    }

    std::lock_guard<std::mutex> lock(m_statsMutex);

    return m_stats.utilizationPercent >
           (m_config.utilizationThreshold * 100.0f);
}

float AdaptiveMemoryPool::calculateFragmentation() const
{
    // Simple fragmentation calculation based on buffer size distribution
    if (m_stats.totalBuffers == 0)
    {
        return 0.0f;
    }

    size_t unusedBuffers = m_stats.availableBuffers;
    float fragmentationRatio =
      static_cast<float>(unusedBuffers) / m_stats.totalBuffers;

    return std::clamp(fragmentationRatio, 0.0f, 1.0f);
}

void AdaptiveMemoryPool::cleanupUnusedBuffers(std::chrono::seconds maxAge)
{
    auto now = std::chrono::steady_clock::now();

    std::lock_guard<std::mutex> lock(m_poolMutex);

    for (size_t i = 0; i < m_bufferInfos.size(); ++i)
    {
        if (m_bufferAvailable[i] && !m_bufferInfos[i].inUse)
        {
            auto age = std::chrono::duration_cast<std::chrono::seconds>(
              now - m_bufferInfos[i].lastUsed
            );
            if (age > maxAge)
            {
                // Mark for removal (simplified - in practice you'd need more
                // sophisticated cleanup)
                m_bufferInfos[i].usageCount =
                  0; // Mark as candidate for removal
            }
        }
    }
}

void AdaptiveMemoryPool::updateStatistics()
{
    std::lock_guard<std::mutex> lock(m_poolMutex);
    std::lock_guard<std::mutex> statsLock(m_statsMutex);

    m_stats.totalBuffers = m_bufferInfos.size();
    m_stats.availableBuffers = 0;
    m_stats.usedBuffers = 0;
    m_stats.totalUsedBytes = 0;

    for (size_t i = 0; i < m_bufferInfos.size(); ++i)
    {
        if (m_bufferAvailable[i])
        {
            m_stats.availableBuffers++;
        }
        else
        {
            m_stats.usedBuffers++;
            m_stats.totalUsedBytes +=
              calculateBufferSize(m_bufferInfos[i].size, m_bufferInfos[i].type);
        }
    }

    m_stats.availableBytes =
      m_stats.totalAllocatedBytes - m_stats.totalUsedBytes;

    if (m_stats.totalAllocatedBytes > 0)
    {
        m_stats.utilizationPercent =
          (static_cast<float>(m_stats.totalUsedBytes) /
           m_stats.totalAllocatedBytes) *
          100.0f;
    }

    m_stats.fragmentationPercent = calculateFragmentation() * 100.0f;
}

std::string AdaptiveMemoryPool::getSizeKey(const cv::Size& size)
{
    return std::to_string(size.width) + "x" + std::to_string(size.height);
}

// BufferHandle implementation
AdaptiveMemoryPool::BufferHandle::BufferHandle(
  AdaptiveMemoryPool* pool, size_t bufferId
)
  : m_pool(pool), m_bufferId(bufferId)
{}

AdaptiveMemoryPool::BufferHandle::~BufferHandle()
{
    release();
}

AdaptiveMemoryPool::BufferHandle::BufferHandle(BufferHandle&& other) noexcept
  : m_pool(other.m_pool), m_bufferId(other.m_bufferId)
{
    other.m_pool = nullptr;
    other.m_bufferId = SIZE_MAX;
}

AdaptiveMemoryPool::BufferHandle& AdaptiveMemoryPool::BufferHandle::operator=(
  BufferHandle&& other
) noexcept
{
    if (this != &other)
    {
        release();
        m_pool = other.m_pool;
        m_bufferId = other.m_bufferId;
        other.m_pool = nullptr;
        other.m_bufferId = SIZE_MAX;
    }
    return *this;
}

#ifdef OPENCV_DNN_CUDA
cv::cuda::GpuMat& AdaptiveMemoryPool::BufferHandle::getGpuMat()
{
    if (!isValid() || m_bufferId >= m_pool->m_gpuBuffers.size())
    {
        throw std::runtime_error("Invalid buffer handle or not a GPU buffer");
    }
    return m_pool->m_gpuBuffers[m_bufferId];
}

const cv::cuda::GpuMat& AdaptiveMemoryPool::BufferHandle::getGpuMat() const
{
    if (!isValid() || m_bufferId >= m_pool->m_gpuBuffers.size())
    {
        throw std::runtime_error("Invalid buffer handle or not a GPU buffer");
    }
    return m_pool->m_gpuBuffers[m_bufferId];
}
#endif

cv::Mat& AdaptiveMemoryPool::BufferHandle::getCpuMat()
{
    if (!isValid() || m_bufferId >= m_pool->m_cpuBuffers.size())
    {
        throw std::runtime_error("Invalid buffer handle or not a CPU buffer");
    }
    return m_pool->m_cpuBuffers[m_bufferId];
}

const cv::Mat& AdaptiveMemoryPool::BufferHandle::getCpuMat() const
{
    if (!isValid() || m_bufferId >= m_pool->m_cpuBuffers.size())
    {
        throw std::runtime_error("Invalid buffer handle or not a CPU buffer");
    }
    return m_pool->m_cpuBuffers[m_bufferId];
}

bool AdaptiveMemoryPool::BufferHandle::isGpu() const
{
    if (!isValid())
    {
        return false;
    }
    return m_pool->m_bufferInfos[m_bufferId].isGpu;
}

cv::Size AdaptiveMemoryPool::BufferHandle::getSize() const
{
    if (!isValid())
    {
        return cv::Size(0, 0);
    }
    return m_pool->m_bufferInfos[m_bufferId].size;
}

int AdaptiveMemoryPool::BufferHandle::getType() const
{
    if (!isValid())
    {
        return -1;
    }
    return m_pool->m_bufferInfos[m_bufferId].type;
}

void AdaptiveMemoryPool::BufferHandle::uploadFrom(const cv::Mat& cpuMat)
{
#ifdef OPENCV_DNN_CUDA
    if (isValid() && isGpu())
    {
        getGpuMat().upload(cpuMat);
    }
#endif
}

void AdaptiveMemoryPool::BufferHandle::downloadTo(cv::Mat& cpuMat) const
{
#ifdef OPENCV_DNN_CUDA
    if (isValid() && isGpu())
    {
        getGpuMat().download(cpuMat);
    }
#endif
}

void AdaptiveMemoryPool::BufferHandle::release()
{
    if (m_pool && m_bufferId != SIZE_MAX)
    {
        m_pool->releaseBuffer(m_bufferId);
        m_pool = nullptr;
        m_bufferId = SIZE_MAX;
    }
}
