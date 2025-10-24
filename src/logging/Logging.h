#ifndef LOGGING_H
#define LOGGING_H

#include "LoggerManager.h"
#include <QString>
#include <string>

/**
 * @brief Convenient logging macros for easy usage throughout the codebase
 *
 * These macros provide a simple interface to the logging system without
 * requiring direct interaction with the LoggerManager or logger interfaces.
 * They automatically convert Qt strings and handle logger retrieval.
 */

namespace logging
{

/**
 * @brief Convert QString to std::string for logging
 */
inline std::string qStringToStdString(const QString& qstr)
{
    return qstr.toStdString();
}

/**
 * @brief Helper function to get logger and handle QString conversion
 */
template <typename... Args>
inline void logMessage(LogLevel level, const QString& format, Args&&... args)
{
    // Safety check for shutdown scenarios - avoid accessing destroyed
    // singletons

    auto& loggerManager = LoggerManager::getInstance();
    if (!loggerManager.isInitialized())
        return;

    auto logger = loggerManager.getLogger();
    if (!logger || !logger->shouldLog(level))
        return;

    std::string stdFormat = qStringToStdString(format);
    switch (level)
    {
    case LogLevel::Trace:
        logger->trace(stdFormat, std::forward<Args>(args)...);
        break;
    case LogLevel::Debug:
        logger->debug(stdFormat, std::forward<Args>(args)...);
        break;
    case LogLevel::Info:
        logger->info(stdFormat, std::forward<Args>(args)...);
        break;
    case LogLevel::Warning:
        logger->warn(stdFormat, std::forward<Args>(args)...);
        break;
    case LogLevel::Error:
        logger->error(stdFormat, std::forward<Args>(args)...);
        break;
    case LogLevel::Critical:
        logger->critical(stdFormat, std::forward<Args>(args)...);
        break;
    default:
        break;
    }
}

/**
 * @brief Helper function for std::string messages
 */
template <typename... Args>
inline void logMessage(
  LogLevel level, const std::string& format, Args&&... args
)
{
    // Safety check for shutdown scenarios - avoid accessing destroyed
    // singletons

    auto& loggerManager = LoggerManager::getInstance();
    if (!loggerManager.isInitialized())
        return;

    auto logger = loggerManager.getLogger();
    if (!logger || !logger->shouldLog(level))
        return;

    switch (level)
    {
    case LogLevel::Trace:
        logger->trace(format, std::forward<Args>(args)...);
        break;
    case LogLevel::Debug:
        logger->debug(format, std::forward<Args>(args)...);
        break;
    case LogLevel::Info:
        logger->info(format, std::forward<Args>(args)...);
        break;
    case LogLevel::Warning:
        logger->warn(format, std::forward<Args>(args)...);
        break;
    case LogLevel::Error:
        logger->error(format, std::forward<Args>(args)...);
        break;
    case LogLevel::Critical:
        logger->critical(format, std::forward<Args>(args)...);
        break;
    default:
        break;
    }
}

/**
 * @brief Helper function for const char* messages
 */
template <typename... Args>
inline void logMessage(LogLevel level, const char* format, Args&&... args)
{
    logMessage(level, std::string(format), std::forward<Args>(args)...);
}

} // namespace logging

// Convenience macros for different log levels
#define LOG_TRACE(...)                                                         \
    logging::logMessage(logging::LogLevel::Trace, __VA_ARGS__)
#define LOG_DEBUG(...)                                                         \
    logging::logMessage(logging::LogLevel::Debug, __VA_ARGS__)
#define LOG_INFO(...) logging::logMessage(logging::LogLevel::Info, __VA_ARGS__)
#define LOG_WARNING(...)                                                       \
    logging::logMessage(logging::LogLevel::Warning, __VA_ARGS__)
#define LOG_ERROR(...)                                                         \
    logging::logMessage(logging::LogLevel::Error, __VA_ARGS__)
#define LOG_CRITICAL(...)                                                      \
    logging::logMessage(logging::LogLevel::Critical, __VA_ARGS__)

// Conditional logging macros (only log if condition is true)
#define LOG_TRACE_IF(condition, ...)                                           \
    do                                                                         \
    {                                                                          \
        if (condition)                                                         \
            LOG_TRACE(__VA_ARGS__);                                            \
    } while (0)
#define LOG_DEBUG_IF(condition, ...)                                           \
    do                                                                         \
    {                                                                          \
        if (condition)                                                         \
            LOG_DEBUG(__VA_ARGS__);                                            \
    } while (0)
#define LOG_INFO_IF(condition, ...)                                            \
    do                                                                         \
    {                                                                          \
        if (condition)                                                         \
            LOG_INFO(__VA_ARGS__);                                             \
    } while (0)
#define LOG_WARNING_IF(condition, ...)                                         \
    do                                                                         \
    {                                                                          \
        if (condition)                                                         \
            LOG_WARNING(__VA_ARGS__);                                          \
    } while (0)
#define LOG_ERROR_IF(condition, ...)                                           \
    do                                                                         \
    {                                                                          \
        if (condition)                                                         \
            LOG_ERROR(__VA_ARGS__);                                            \
    } while (0)
#define LOG_CRITICAL_IF(condition, ...)                                        \
    do                                                                         \
    {                                                                          \
        if (condition)                                                         \
            LOG_CRITICAL(__VA_ARGS__);                                         \
    } while (0)

// Macros for named loggers
#define LOG_NAMED_TRACE(name, ...)                                             \
    do                                                                         \
    {                                                                          \
        auto logger = logging::LoggerManager::getInstance().getLogger(name);   \
        if (logger && logger->shouldLog(logging::LogLevel::Trace))             \
            logger->trace(__VA_ARGS__);                                        \
    } while (0)

#define LOG_NAMED_DEBUG(name, ...)                                             \
    do                                                                         \
    {                                                                          \
        auto logger = logging::LoggerManager::getInstance().getLogger(name);   \
        if (logger && logger->shouldLog(logging::LogLevel::Debug))             \
            logger->debug(__VA_ARGS__);                                        \
    } while (0)

#define LOG_NAMED_INFO(name, ...)                                              \
    do                                                                         \
    {                                                                          \
        auto logger = logging::LoggerManager::getInstance().getLogger(name);   \
        if (logger && logger->shouldLog(logging::LogLevel::Info))              \
            logger->info(__VA_ARGS__);                                         \
    } while (0)

#define LOG_NAMED_WARNING(name, ...)                                           \
    do                                                                         \
    {                                                                          \
        auto logger = logging::LoggerManager::getInstance().getLogger(name);   \
        if (logger && logger->shouldLog(logging::LogLevel::Warning))           \
            logger->warn(__VA_ARGS__);                                         \
    } while (0)

#define LOG_NAMED_ERROR(name, ...)                                             \
    do                                                                         \
    {                                                                          \
        auto logger = logging::LoggerManager::getInstance().getLogger(name);   \
        if (logger && logger->shouldLog(logging::LogLevel::Error))             \
            logger->error(__VA_ARGS__);                                        \
    } while (0)

#define LOG_NAMED_CRITICAL(name, ...)                                          \
    do                                                                         \
    {                                                                          \
        auto logger = logging::LoggerManager::getInstance().getLogger(name);   \
        if (logger && logger->shouldLog(logging::LogLevel::Critical))          \
            logger->critical(__VA_ARGS__);                                     \
    } while (0)

// Utility macros
#define LOG_FLUSH() logging::LoggerManager::getInstance().flushAll()
#define LOG_SET_LEVEL(level)                                                   \
    logging::LoggerManager::getInstance().setGlobalLevel(level)

// Compatibility macros for migration from Qt logging
#define qLog(...) LOG_DEBUG(__VA_ARGS__) // Replaces qDebug() calls
#define qLogInfo(...) LOG_INFO(__VA_ARGS__)
#define qLogWarn(...) LOG_WARNING(__VA_ARGS__)
#define qLogError(...) LOG_ERROR(__VA_ARGS__)
#define qLogCritical(...) LOG_CRITICAL(__VA_ARGS__)

#endif // LOGGING_H
