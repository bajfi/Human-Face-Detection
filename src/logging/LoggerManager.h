#ifndef LOGGERMANAGER_H
#define LOGGERMANAGER_H

#include "ILogger.h"
#include "SpdlogLogger.h"
#include <memory>
#include <string>
#include <unordered_map>
#include <mutex>

namespace logging
{

/**
 * @brief Singleton manager for application-wide logging
 *
 * The LoggerManager provides centralized management of loggers throughout
 * the application. It handles logger creation, configuration, and lifecycle.
 * This class follows the dependency injection principle by providing
 * interface-based access to logging functionality.
 */
class LoggerManager
{
  public:
    /**
     * @brief Configuration for the LoggerManager
     */
    struct Config
    {
        LogLevel defaultLevel = LogLevel::Info;
        std::string defaultPattern = "[%Y-%m-%d %H:%M:%S.%e] [%n] [%l] %v";
        bool enableConsoleLogging = true;
        bool enableFileLogging = false;
        std::string logDirectory = "logs";
        std::string defaultLogFile = "";
        size_t maxFileSize = 1024 * 1024 * 5; // 5MB
        size_t maxFiles = 3;
        bool enableAsync = false;
    };

    /**
     * @brief Get the singleton instance
     */
    static LoggerManager& getInstance();

    /**
     * @brief Initialize the logging system with configuration
     */
    void initialize(const Config& config);

    /**
     * @brief Initialize the logging system with default configuration
     */
    void initialize();

    /**
     * @brief Get the default application logger
     */
    LoggerPtr getLogger();

    /**
     * @brief Get a named logger (creates if doesn't exist)
     */
    LoggerPtr getLogger(const std::string& name);

    /**
     * @brief Create a logger with specific configuration
     */
    LoggerPtr createLogger(
      const std::string& name, const SpdlogLogger::Config& config
    );

    /**
     * @brief Set global log level for all loggers
     */
    void setGlobalLevel(LogLevel level);

    /**
     * @brief Get current global log level
     */
    LogLevel getGlobalLevel() const;

    /**
     * @brief Flush all loggers
     */
    void flushAll();

    /**
     * @brief Shutdown logging system (call on application exit)
     */
    void shutdown();

    /**
     * @brief Check if logging system is initialized
     */
    bool isInitialized() const
    {
        return m_initialized;
    }

    /**
     * @brief Get configuration
     */
    const Config& getConfig() const
    {
        return m_config;
    }

    // Delete copy constructor and assignment operator
    LoggerManager(const LoggerManager&) = delete;
    LoggerManager& operator=(const LoggerManager&) = delete;

  private:
    LoggerManager() = default;
    ~LoggerManager();

    void createDefaultLogger();
    std::string generateLogFilename(const std::string& loggerName) const;

    mutable std::mutex m_mutex;
    std::unordered_map<std::string, LoggerPtr> m_loggers;
    LoggerPtr m_defaultLogger;
    Config m_config;
    bool m_initialized = false;
    bool m_shutdown = false;
    LogLevel m_globalLevel = LogLevel::Info;

    static constexpr const char* DEFAULT_LOGGER_NAME = "qtapp";
};

} // namespace logging

// Convenience macros for easy access to the default logger
#define LOG_MANAGER() utils::logging::LoggerManager::getInstance()
#define LOG_DEFAULT() utils::logging::LoggerManager::getInstance().getLogger()

#endif // LOGGERMANAGER_H
