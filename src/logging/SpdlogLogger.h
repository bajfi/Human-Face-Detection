#ifndef SPDLOGLOGGER_H
#define SPDLOGLOGGER_H

#include "ILogger.h"
#include <spdlog/spdlog.h>
#include <spdlog/sinks/stdout_color_sinks.h>
#include <spdlog/sinks/basic_file_sink.h>
#include <spdlog/sinks/rotating_file_sink.h>
#include <memory>
#include <string>
#include <cstdarg>

namespace logging
{

/**
 * @brief Concrete implementation of ILogger using spdlog library
 *
 * This class wraps spdlog functionality and provides the interface
 * defined by ILogger. It supports console and file logging with
 * configurable log levels and formats.
 */
class SpdlogLogger : public ILogger
{
  public:
    /**
     * @brief Configuration structure for SpdlogLogger
     */
    struct Config
    {
        std::string name = "default";
        LogLevel level = LogLevel::Info;
        std::string pattern = "[%Y-%m-%d %H:%M:%S.%e] [%n] [%l] %v";
        bool enableConsole = true;
        bool enableFile = false;
        std::string filename = "";
        size_t maxFileSize = 1024 * 1024 * 5; // 5MB
        size_t maxFiles = 3;
        bool enableAsync = false;
    };

    /**
     * @brief Construct logger with configuration
     */
    explicit SpdlogLogger(const Config& config);

    /**
     * @brief Construct logger with default configuration
     */
    SpdlogLogger();

    /**
     * @brief Construct logger with name and level
     */
    SpdlogLogger(const std::string& name, LogLevel level = LogLevel::Info);

    ~SpdlogLogger() override = default;

    // Core logging methods
    void trace(const std::string& message) override;
    void debug(const std::string& message) override;
    void info(const std::string& message) override;
    void warn(const std::string& message) override;
    void error(const std::string& message) override;
    void critical(const std::string& message) override;

    // Configuration methods
    void setLevel(LogLevel level) override;
    LogLevel getLevel() const override;
    bool shouldLog(LogLevel level) const override;

    // Logger management
    void flush() override;
    std::string getName() const override;

    // Additional configuration methods
    void setPattern(const std::string& pattern);
    void addFileSink(
      const std::string& filename,
      size_t maxFileSize = 1024 * 1024 * 5,
      size_t maxFiles = 3
    );
    void addConsoleSink(bool colorEnabled = true);

    // Get the underlying spdlog logger for advanced usage
    std::shared_ptr<spdlog::logger> getSpdlogLogger() const
    {
        return m_logger;
    }

  protected:
    // Protected methods for formatted logging
    void traceFormatted(const std::string& format, ...) override;
    void debugFormatted(const std::string& format, ...) override;
    void infoFormatted(const std::string& format, ...) override;
    void warnFormatted(const std::string& format, ...) override;
    void errorFormatted(const std::string& format, ...) override;
    void criticalFormatted(const std::string& format, ...) override;

  private:
    void initializeLogger(const Config& config);
    spdlog::level::level_enum convertLogLevel(LogLevel level) const;
    LogLevel convertSpdlogLevel(spdlog::level::level_enum level) const;
    std::string formatMessage(const std::string& format, va_list args) const;

    std::shared_ptr<spdlog::logger> m_logger;
    std::string m_name;
    LogLevel m_currentLevel;
};

} // namespace logging

#endif // SPDLOGLOGGER_H
