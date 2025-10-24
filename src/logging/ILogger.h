#ifndef ILOGGER_H
#define ILOGGER_H

#include <memory>
#include <string>

namespace logging
{

/**
 * @brief Log level enumeration
 */
enum class LogLevel
{
    Trace = 0,
    Debug = 1,
    Info = 2,
    Warning = 3,
    Error = 4,
    Critical = 5,
    Off = 6
};

/**
 * @brief Abstract logging interface to decouple from specific logging
 * implementations
 *
 * This interface provides a clean abstraction over logging functionality,
 * allowing the application to switch between different logging backends
 * (spdlog, Qt logging, custom implementations, etc.) without changing client
 * code.
 */
class ILogger
{
  public:
    virtual ~ILogger() = default;

    // Core logging methods
    virtual void trace(const std::string& message) = 0;
    virtual void debug(const std::string& message) = 0;
    virtual void info(const std::string& message) = 0;
    virtual void warn(const std::string& message) = 0;
    virtual void error(const std::string& message) = 0;
    virtual void critical(const std::string& message) = 0;

    // Template methods for formatted logging
    template <typename... Args>
    void trace(const std::string& format, Args&&... args)
    {
        if (shouldLog(LogLevel::Trace))
        {
            traceFormatted(format, std::forward<Args>(args)...);
        }
    }

    template <typename... Args>
    void debug(const std::string& format, Args&&... args)
    {
        if (shouldLog(LogLevel::Debug))
        {
            debugFormatted(format, std::forward<Args>(args)...);
        }
    }

    template <typename... Args>
    void info(const std::string& format, Args&&... args)
    {
        if (shouldLog(LogLevel::Info))
        {
            infoFormatted(format, std::forward<Args>(args)...);
        }
    }

    template <typename... Args>
    void warn(const std::string& format, Args&&... args)
    {
        if (shouldLog(LogLevel::Warning))
        {
            warnFormatted(format, std::forward<Args>(args)...);
        }
    }

    template <typename... Args>
    void error(const std::string& format, Args&&... args)
    {
        if (shouldLog(LogLevel::Error))
        {
            errorFormatted(format, std::forward<Args>(args)...);
        }
    }

    template <typename... Args>
    void critical(const std::string& format, Args&&... args)
    {
        if (shouldLog(LogLevel::Critical))
        {
            criticalFormatted(format, std::forward<Args>(args)...);
        }
    }

    // Configuration methods
    virtual void setLevel(LogLevel level) = 0;
    virtual LogLevel getLevel() const = 0;
    virtual bool shouldLog(LogLevel level) const = 0;

    // Logger management
    virtual void flush() = 0;
    virtual std::string getName() const = 0;

  protected:
    // Protected methods for formatted logging - implemented by concrete classes
    virtual void traceFormatted(const std::string& format, ...) = 0;
    virtual void debugFormatted(const std::string& format, ...) = 0;
    virtual void infoFormatted(const std::string& format, ...) = 0;
    virtual void warnFormatted(const std::string& format, ...) = 0;
    virtual void errorFormatted(const std::string& format, ...) = 0;
    virtual void criticalFormatted(const std::string& format, ...) = 0;
};

using LoggerPtr = std::shared_ptr<ILogger>;

/**
 * @brief Convert log level to string
 */
inline std::string logLevelToString(LogLevel level)
{
    switch (level)
    {
    case LogLevel::Trace:
        return "trace";
    case LogLevel::Debug:
        return "debug";
    case LogLevel::Info:
        return "info";
    case LogLevel::Warning:
        return "warning";
    case LogLevel::Error:
        return "error";
    case LogLevel::Critical:
        return "critical";
    case LogLevel::Off:
        return "off";
    default:
        return "unknown";
    }
}

/**
 * @brief Convert string to log level
 */
inline LogLevel stringToLogLevel(const std::string& levelStr)
{
    if (levelStr == "trace")
        return LogLevel::Trace;
    if (levelStr == "debug")
        return LogLevel::Debug;
    if (levelStr == "info")
        return LogLevel::Info;
    if (levelStr == "warning" || levelStr == "warn")
        return LogLevel::Warning;
    if (levelStr == "error")
        return LogLevel::Error;
    if (levelStr == "critical")
        return LogLevel::Critical;
    if (levelStr == "off")
        return LogLevel::Off;

    return LogLevel::Info; // Default fallback
}

} // namespace logging

#endif // ILOGGER_H
