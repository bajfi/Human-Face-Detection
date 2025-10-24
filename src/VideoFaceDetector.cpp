// Copyright (c) 2025 JackLee
//
// This software is released under the MIT License.
// https://opensource.org/licenses/MIT

#include "VideoFaceDetector.h"
#include <algorithm>
#include <cmath>
#include <iostream>

VideoFaceDetector::VideoFaceDetector(std::shared_ptr<IFaceDetector> detector)
  : m_detector(std::move(detector)),
    m_memoryPool(std::make_unique<VideoProcessingPool>())
{
    if (!m_detector)
    {
        throw std::invalid_argument("Face detector cannot be null");
    }

    m_lastFrameTime = std::chrono::high_resolution_clock::now();
}

VideoFaceDetector::~VideoFaceDetector()
{
    stopProcessing();
    cleanup();
}

bool VideoFaceDetector::initialize(const ProcessingConfig& config)
{
    std::lock_guard<std::mutex> lock(m_configMutex);

    if (m_processing.load())
    {
        return false; // Cannot reconfigure while processing
    }

    m_config = config;

    // Validate configuration
    m_config.frameSkipCount = std::max(0, m_config.frameSkipCount);

    m_config.maxQueueSize =
      std::max(1, m_config.maxQueueSize); // At least 1 frame in queue

    if (m_config.targetResolution.width < 64 ||
        m_config.targetResolution.height < 64)
    {
        m_config.targetResolution = cv::Size(640, 480);
    }

    // Initialize memory pool with target resolution
    if (!initializeMemoryPool(m_config.targetResolution))
    {
        std::cerr << "Warning: Failed to initialize GPU memory pool, falling "
                     "back to CPU"
                  << std::endl;
        m_config.gpuAcceleration = false;
    }

    // Configure the underlying detector
    if (m_detector)
    {
        m_detector->setDetectionParams(1.1, 3, cv::Size(30, 30), cv::Size());
    }

    return true;
}

bool VideoFaceDetector::startProcessing(
  cv::VideoCapture& capture, FrameCallback callback
)
{
    if (m_processing.load() || !m_detector || !callback)
    {
        return false;
    }

    if (!capture.isOpened())
    {
        return false;
    }

    m_frameCallback = callback;
    m_stopRequested.store(false, std::memory_order_relaxed);
    m_processing.store(true, std::memory_order_relaxed);

    // Reset statistics
    {
        std::lock_guard<std::mutex> lock(m_statsMutex);
        m_stats = ProcessingStats{};
        m_stats.gpuAccelerationActive =
          m_config.gpuAcceleration && m_memoryPool->initialized;
    }

    // Start processing thread if async processing is enabled
    if (m_config.asyncProcessing)
    {
        m_processingThread = std::make_unique<std::thread>(
          &VideoFaceDetector::processingThreadFunc, this
        );
    }

    // Main video capture loop
    cv::Mat frame;
    auto lastFrameTime = std::chrono::high_resolution_clock::now();

    while (m_processing.load() && !m_stopRequested.load())
    {
        if (!capture.read(frame))
        {
            break; // End of stream or error
        }

        if (frame.empty())
        {
            continue;
        }

        auto currentTime = std::chrono::high_resolution_clock::now();
        auto frameInterval =
          std::chrono::duration_cast<std::chrono::milliseconds>(
            currentTime - lastFrameTime
          );
        lastFrameTime = currentTime;

        if (shouldSkipFrame())
        {
            std::lock_guard<std::mutex> lock(m_statsMutex);
            m_stats.skippedFrames++;
            continue;
        }

        if (m_config.asyncProcessing)
        {
            // Add frame to queue for async processing
            {
                std::lock_guard<std::mutex> lock(m_queueMutex);
                if (m_frameQueue.size() >=
                    static_cast<size_t>(m_config.maxQueueSize))
                {
                    // Drop oldest frame if queue is full
                    m_frameQueue.pop();
                    std::lock_guard<std::mutex> statsLock(m_statsMutex);
                    m_stats.droppedFrames++;
                }
                m_frameQueue.push(frame.clone());
            }
            m_queueCondition.notify_one();
        }
        else
        {
            // Synchronous processing
            auto result = processSingleFrame(frame);
            if (m_frameCallback)
            {
                m_frameCallback(result);
            }
        }
    }

    return true;
}

void VideoFaceDetector::stopProcessing()
{
    m_stopRequested.store(true, std::memory_order_release);
    m_processing.store(false, std::memory_order_release);

    // Wake up processing thread
    m_queueCondition.notify_all();

    // Wait for processing thread to finish
    if (m_processingThread && m_processingThread->joinable())
    {
        m_processingThread->join();
        m_processingThread.reset();
    }

    // Clear frame queue
    {
        std::lock_guard<std::mutex> lock(m_queueMutex);
        while (!m_frameQueue.empty())
        {
            m_frameQueue.pop();
        }
    }
}

VideoFaceDetector::FrameResult VideoFaceDetector::processSingleFrame(
  const cv::Mat& frame
)
{
    FrameResult result;
    result.frameIndex = m_frameCounter.fetch_add(1);

    auto startTime = std::chrono::high_resolution_clock::now();

    if (frame.empty() || !m_detector)
    {
        return result;
    }

    try
    {
        // Choose processing path based on GPU availability and configuration
        if (m_config.gpuAcceleration && m_memoryPool->initialized)
        {
            result = processFrameGpu(frame);
        }
        else
        {
            result = processFrameCpu(frame);
        }

        auto endTime = std::chrono::high_resolution_clock::now();
        result.processingTime =
          std::chrono::duration_cast<std::chrono::milliseconds>(
            endTime - startTime
          );

        // Update statistics
        updateStats(result.processingTime);

        // Adaptive quality control
        if (m_config.adaptiveQuality)
        {
            adaptQuality();
        }
    }
    catch (const std::exception& e)
    {
        std::cerr << "Error processing frame: " << e.what() << std::endl;
    }

    return result;
}

void VideoFaceDetector::enableAsyncProcessing(bool enable)
{
    std::lock_guard<std::mutex> lock(m_configMutex);
    m_config.asyncProcessing = enable;
}

void VideoFaceDetector::setFrameSkipping(int skipCount)
{
    std::lock_guard<std::mutex> lock(m_configMutex);
    m_config.frameSkipCount = std::max(0, skipCount);
}

bool VideoFaceDetector::enableGpuAcceleration(bool enable)
{
    std::lock_guard<std::mutex> lock(m_configMutex);

#ifdef OPENCV_DNN_CUDA
    if (enable && cv::cuda::getCudaEnabledDeviceCount() > 0)
    {
        m_config.gpuAcceleration = true;
        if (!m_memoryPool->initialized)
        {
            initializeMemoryPool(m_config.targetResolution);
        }
        return m_memoryPool->initialized;
    }
    else
    {
        m_config.gpuAcceleration = false;
        return false;
    }
#else
    m_config.gpuAcceleration = false;
    return false;
#endif
}

void VideoFaceDetector::setTargetResolution(const cv::Size& resolution)
{
    std::lock_guard<std::mutex> lock(m_configMutex);
    if (resolution.width >= 64 && resolution.height >= 64)
    {
        m_config.targetResolution = resolution;

        // Reinitialize memory pool if processing is not active
        if (!m_processing.load(std::memory_order_acquire))
        {
            initializeMemoryPool(resolution);
        }
    }
}

void VideoFaceDetector::setConfidenceThreshold(float threshold)
{
    std::lock_guard<std::mutex> lock(m_configMutex);
    m_config.confidenceThreshold = std::clamp(threshold, 0.0f, 1.0f);
}

void VideoFaceDetector::enableAdaptiveQuality(bool enable)
{
    std::lock_guard<std::mutex> lock(m_configMutex);
    m_config.adaptiveQuality = enable;
}

VideoFaceDetector::ProcessingStats VideoFaceDetector::getProcessingStats() const
{
    std::lock_guard<std::mutex> lock(m_statsMutex);
    return m_stats;
}

bool VideoFaceDetector::isProcessing() const
{
    return m_processing.load(std::memory_order_acquire);
}

bool VideoFaceDetector::isGpuAccelerationEnabled() const
{
    std::lock_guard<std::mutex> lock(m_configMutex);
    return m_config.gpuAcceleration && m_memoryPool->initialized;
}

VideoFaceDetector::ProcessingConfig VideoFaceDetector::getConfiguration() const
{
    std::lock_guard<std::mutex> lock(m_configMutex);
    return m_config;
}

bool VideoFaceDetector::updateConfiguration(const ProcessingConfig& config)
{
    if (m_processing.load(std::memory_order_acquire))
    {
        return false; // Cannot update configuration while processing
    }

    return initialize(config);
}

VideoFaceDetector::ProcessingConfig VideoFaceDetector::getRecommendedConfig(
  bool prioritizeSpeed
)
{
    ProcessingConfig config;

#ifdef OPENCV_DNN_CUDA
    bool cudaAvailable = cv::cuda::getCudaEnabledDeviceCount() > 0;
#else
    bool cudaAvailable = false;
#endif

    if (prioritizeSpeed)
    {
        config.frameSkipCount =
          cudaAvailable ? 0 : 1; // Skip every other frame on CPU
        config.targetResolution =
          cudaAvailable ? cv::Size(640, 480) : cv::Size(320, 240);
        config.confidenceThreshold =
          0.7f; // Higher threshold for fewer false positives
        config.asyncProcessing = true;
        config.gpuAcceleration = cudaAvailable;
        config.adaptiveQuality = true;
        config.maxQueueSize = 5;
    }
    else
    {
        // Prioritize accuracy
        config.frameSkipCount = 0;      // Process all frames
        config.targetResolution = cv::Size(800, 600);
        config.confidenceThreshold =
          0.5f;                         // Lower threshold for more detections
        config.asyncProcessing = true;
        config.gpuAcceleration = cudaAvailable;
        config.adaptiveQuality = false; // Consistent quality
        config.maxQueueSize = 10;
    }

    return config;
}

bool VideoFaceDetector::initializeMemoryPool(const cv::Size& frameSize)
{
    if (!m_memoryPool)
    {
        m_memoryPool = std::make_unique<VideoProcessingPool>();
    }

#ifdef OPENCV_DNN_CUDA
    try
    {
        if (cv::cuda::getCudaEnabledDeviceCount() > 0)
        {
            // Initialize GPU buffers
            for (size_t i = 0; i < VideoProcessingPool::BUFFER_COUNT; ++i)
            {
                m_memoryPool->frameBuffers[i].create(frameSize, CV_8UC3);
            }

            m_memoryPool->preprocessedFrame.create(frameSize, CV_8UC3);
            m_memoryPool->resizedFrame.create(frameSize, CV_8UC3);

            m_memoryPool->bufferSize = frameSize;
            m_memoryPool->initialized = true;

            std::cout << "GPU memory pool initialized: " << frameSize.width
                      << "x" << frameSize.height << std::endl;
            return true;
        }
    }
    catch (const cv::Exception& e)
    {
        std::cerr << "Failed to initialize GPU memory pool: " << e.what()
                  << std::endl;
    }
#endif

    // Fallback to CPU buffers
    for (size_t i = 0; i < VideoProcessingPool::BUFFER_COUNT; ++i)
    {
        m_memoryPool->cpuBuffers[i] = cv::Mat::zeros(frameSize, CV_8UC3);
    }

    m_memoryPool->bufferSize = frameSize;
    m_memoryPool->initialized = true;

    std::cout << "CPU memory pool initialized: " << frameSize.width << "x"
              << frameSize.height << std::endl;
    return true;
}

void VideoFaceDetector::processingThreadFunc()
{
    while (m_processing.load(std::memory_order_acquire) &&
           !m_stopRequested.load(std::memory_order_acquire))
    {
        cv::Mat frame;

        // Wait for frame in queue
        {
            std::unique_lock<std::mutex> lock(m_queueMutex);
            m_queueCondition.wait(
              lock,
              [this]
              {
                  return !m_frameQueue.empty() ||
                         m_stopRequested.load(std::memory_order_acquire);
              }
            );

            if (m_stopRequested.load(std::memory_order_acquire))
            {
                break;
            }

            if (!m_frameQueue.empty())
            {
                frame = m_frameQueue.front();
                m_frameQueue.pop();
            }
        }

        if (!frame.empty())
        {
            auto result = processSingleFrame(frame);
            if (m_frameCallback)
            {
                m_frameCallback(result);
            }
        }
    }
}

VideoFaceDetector::FrameResult VideoFaceDetector::processFrameGpu(
  const cv::Mat& frame
)
{
    FrameResult result;
    result.processedFrame = frame;

#ifdef OPENCV_DNN_CUDA
    try
    {
        // Get current buffer index
        size_t bufferIndex =
          m_memoryPool->currentBufferIndex.load(std::memory_order_acquire);

        // Upload frame to GPU
        m_memoryPool->frameBuffers[bufferIndex].upload(
          frame, m_memoryPool->asyncStream
        );

        // Resize if necessary
        cv::cuda::GpuMat* processBuffer =
          &m_memoryPool->frameBuffers[bufferIndex];

        if (frame.size() != m_config.targetResolution)
        {
            // Use CPU resize for now - OpenCV CUDA resize might not be
            // available in all builds
            cv::Mat tempFrame;
            processBuffer->download(tempFrame);
            cv::resize(tempFrame, tempFrame, m_config.targetResolution);
            m_memoryPool->resizedFrame.upload(tempFrame);

            processBuffer = &m_memoryPool->resizedFrame;
        }

        // Download processed frame back to CPU for detection
        cv::Mat cpuFrame;
        processBuffer->download(cpuFrame, m_memoryPool->asyncStream);
        m_memoryPool->asyncStream.waitForCompletion();

        // Perform face detection on CPU (most detectors don't have full GPU
        // support yet)
        result.detections = m_detector->detectFaces(cpuFrame);
        result.processedFrame = cpuFrame;

        // Update buffer index for triple buffering
        m_memoryPool->currentBufferIndex.store(
          (bufferIndex + 1) % VideoProcessingPool::BUFFER_COUNT
        );
    }
    catch (const cv::Exception& e)
    {
        std::cerr << "GPU processing error, falling back to CPU: " << e.what()
                  << std::endl;
        return processFrameCpu(frame);
    }
#else
    return processFrameCpu(frame);
#endif

    return result;
}

VideoFaceDetector::FrameResult VideoFaceDetector::processFrameCpu(
  const cv::Mat& frame
)
{
    FrameResult result;

    // Resize frame if necessary
    cv::Mat processFrame;
    if (frame.size() != m_config.targetResolution)
    {
        cv::resize(frame, processFrame, m_config.targetResolution);
    }
    else
    {
        processFrame = frame;
    }

    // Perform face detection
    result.detections = m_detector->detectFaces(processFrame);
    result.processedFrame = processFrame;

    return result;
}

void VideoFaceDetector::updateStats(std::chrono::milliseconds processingTime)
{
    std::lock_guard<std::mutex> lock(m_statsMutex);

    m_stats.processedFrames++;
    m_stats.lastFrameTime = processingTime;

    // Update rolling average
    if (m_stats.processedFrames == 1)
    {
        m_stats.avgProcessingTime = processingTime;
    }
    else
    {
        // Exponential moving average with alpha = 0.1
        auto avgMs = m_stats.avgProcessingTime.count();
        avgMs = static_cast<long>(0.9 * avgMs + 0.1 * processingTime.count());
        m_stats.avgProcessingTime = std::chrono::milliseconds(avgMs);
    }

    // Update FPS calculation
    auto currentTime = std::chrono::high_resolution_clock::now();
    auto timeDiff = std::chrono::duration_cast<std::chrono::milliseconds>(
      currentTime - m_lastFrameTime
    );

    if (auto timeDiffCount = timeDiff.count(); timeDiffCount > 0)
    {
        m_stats.currentFPS = 1000.0 / timeDiffCount;
    }

    m_lastFrameTime = currentTime;

    // Update GPU memory usage estimate
    m_stats.gpuMemoryUsage = estimateGpuMemoryUsage();
}

bool VideoFaceDetector::shouldSkipFrame() const
{
    if (m_config.frameSkipCount == 0)
    {
        return false;
    }

    return (m_frameCounter.load(std::memory_order_acquire) %
            (m_config.frameSkipCount + 1)) != 0;
}

void VideoFaceDetector::adaptQuality()
{
    std::lock_guard<std::mutex> lock(m_statsMutex);

    // If processing is too slow, adapt quality
    if (m_stats.avgProcessingTime > ProcessingStats::TARGET_FRAME_TIME)
    {
        // Increase frame skipping
        if (m_config.frameSkipCount < 3)
        {
            m_config.frameSkipCount++;
            std::cout << "Performance adaptation: Increased frame skipping to "
                      << m_config.frameSkipCount << std::endl;
        }

        // Reduce resolution if still too slow
        if (m_stats.avgProcessingTime > ProcessingStats::TARGET_FRAME_TIME * 2)
        {
            if (m_config.targetResolution.width > 320)
            {
                m_config.targetResolution.width = std::max(
                  320, static_cast<int>(m_config.targetResolution.width * 0.8)
                );
                m_config.targetResolution.height = std::max(
                  240, static_cast<int>(m_config.targetResolution.height * 0.8)
                );
                std::cout << "Performance adaptation: Reduced resolution to "
                          << m_config.targetResolution.width << "x"
                          << m_config.targetResolution.height << std::endl;
            }
        }
    }
    else if (m_stats.avgProcessingTime < ProcessingStats::TARGET_FRAME_TIME / 2)
    {
        // Performance is good, can increase quality
        if (m_config.frameSkipCount > 0)
        {
            m_config.frameSkipCount--;
            std::cout << "Performance adaptation: Reduced frame skipping to "
                      << m_config.frameSkipCount << std::endl;
        }
    }
}

void VideoFaceDetector::cleanup()
{
    stopProcessing();

    if (m_memoryPool)
    {
#ifdef OPENCV_DNN_CUDA
        // CUDA resources are automatically released by OpenCV
#endif
        m_memoryPool.reset();
    }
}

size_t VideoFaceDetector::estimateGpuMemoryUsage() const
{
    if (!m_config.gpuAcceleration || !m_memoryPool->initialized)
    {
        return 0;
    }

    size_t bytesPerPixel = 3; // BGR format
    size_t frameSize = m_config.targetResolution.width *
                       m_config.targetResolution.height * bytesPerPixel;

    // Triple buffering + preprocessing buffers
    size_t totalUsage = frameSize * (VideoProcessingPool::BUFFER_COUNT + 2);

    return totalUsage;
}
