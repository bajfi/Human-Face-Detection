#include "SpdlogLogger.h"
#include <spdlog/sinks/stdout_color_sinks.h>
#include <spdlog/sinks/basic_file_sink.h>
#include <spdlog/sinks/rotating_file_sink.h>
#include <vector>
#include <cstdio>
#include <cstring>

namespace logging
{

SpdlogLogger::SpdlogLogger(const Config& config)
  : m_name(config.name), m_currentLevel(config.level)
{
    initializeLogger(config);
}

SpdlogLogger::SpdlogLogger() : m_name("default"), m_currentLevel(LogLevel::Info)
{
    Config config;
    initializeLogger(config);
}

SpdlogLogger::SpdlogLogger(const std::string& name, LogLevel level)
  : m_name(name), m_currentLevel(level)
{
    Config config;
    config.name = name;
    config.level = level;
    initializeLogger(config);
}

void SpdlogLogger::initializeLogger(const Config& config)
{
    std::vector<spdlog::sink_ptr> sinks;

    // Add console sink if enabled
    if (config.enableConsole)
    {
        auto consoleSink =
          std::make_shared<spdlog::sinks::stdout_color_sink_mt>();
        consoleSink->set_level(convertLogLevel(config.level));
        sinks.push_back(consoleSink);
    }

    // Add file sink if enabled
    if (config.enableFile && !config.filename.empty())
    {
        try
        {
            auto fileSink =
              std::make_shared<spdlog::sinks::rotating_file_sink_mt>(
                config.filename, config.maxFileSize, config.maxFiles
              );
            fileSink->set_level(convertLogLevel(config.level));
            sinks.push_back(fileSink);
        }
        catch (const spdlog::spdlog_ex& ex)
        {
            // Fallback to console only if file sink fails
            if (sinks.empty())
            {
                auto consoleSink =
                  std::make_shared<spdlog::sinks::stdout_color_sink_mt>();
                consoleSink->set_level(convertLogLevel(config.level));
                sinks.push_back(consoleSink);
            }
        }
    }

    // Create logger with sinks
    if (sinks.empty())
    {
        // Default to console sink if no sinks configured
        auto consoleSink =
          std::make_shared<spdlog::sinks::stdout_color_sink_mt>();
        consoleSink->set_level(convertLogLevel(config.level));
        sinks.push_back(consoleSink);
    }

    m_logger =
      std::make_shared<spdlog::logger>(config.name, sinks.begin(), sinks.end());
    m_logger->set_level(convertLogLevel(config.level));
    m_logger->set_pattern(config.pattern);

    // Register logger with spdlog
    spdlog::register_logger(m_logger);
}

void SpdlogLogger::trace(const std::string& message)
{
    if (m_logger)
    {
        m_logger->trace(message);
    }
}

void SpdlogLogger::debug(const std::string& message)
{
    if (m_logger)
    {
        m_logger->debug(message);
    }
}

void SpdlogLogger::info(const std::string& message)
{
    if (m_logger)
    {
        m_logger->info(message);
    }
}

void SpdlogLogger::warn(const std::string& message)
{
    if (m_logger)
    {
        m_logger->warn(message);
    }
}

void SpdlogLogger::error(const std::string& message)
{
    if (m_logger)
    {
        m_logger->error(message);
    }
}

void SpdlogLogger::critical(const std::string& message)
{
    if (m_logger)
    {
        m_logger->critical(message);
    }
}

void SpdlogLogger::setLevel(LogLevel level)
{
    m_currentLevel = level;
    if (m_logger)
    {
        m_logger->set_level(convertLogLevel(level));
    }
}

LogLevel SpdlogLogger::getLevel() const
{
    return m_currentLevel;
}

bool SpdlogLogger::shouldLog(LogLevel level) const
{
    return static_cast<int>(level) >= static_cast<int>(m_currentLevel);
}

void SpdlogLogger::flush()
{
    if (m_logger)
    {
        m_logger->flush();
    }
}

std::string SpdlogLogger::getName() const
{
    return m_name;
}

void SpdlogLogger::setPattern(const std::string& pattern)
{
    if (m_logger)
    {
        m_logger->set_pattern(pattern);
    }
}

void SpdlogLogger::addFileSink(
  const std::string& filename, size_t maxFileSize, size_t maxFiles
)
{
    if (!m_logger)
        return;

    try
    {
        auto fileSink = std::make_shared<spdlog::sinks::rotating_file_sink_mt>(
          filename, maxFileSize, maxFiles
        );
        fileSink->set_level(convertLogLevel(m_currentLevel));

        // Create new logger with existing sinks plus new file sink
        auto sinks = m_logger->sinks();
        sinks.push_back(fileSink);

        auto newLogger =
          std::make_shared<spdlog::logger>(m_name, sinks.begin(), sinks.end());
        newLogger->set_level(convertLogLevel(m_currentLevel));
        newLogger->set_pattern("[%Y-%m-%d %H:%M:%S.%e] [%n] [%l] %v");

        // Replace the logger
        spdlog::drop(m_name);
        m_logger = newLogger;
        spdlog::register_logger(m_logger);
    }
    catch (const spdlog::spdlog_ex& ex)
    {
        // Log error to current logger if possible
        if (m_logger)
        {
            m_logger->error(
              "Failed to add file sink '{}': {}", filename, ex.what()
            );
        }
    }
}

void SpdlogLogger::addConsoleSink(bool colorEnabled)
{
    if (!m_logger)
        return;

    auto consoleSink =
      colorEnabled ? std::static_pointer_cast<spdlog::sinks::sink>(
                       std::make_shared<spdlog::sinks::stdout_color_sink_mt>()
                     )
                   : std::static_pointer_cast<spdlog::sinks::sink>(
                       std::make_shared<spdlog::sinks::stdout_color_sink_mt>()
                     );

    consoleSink->set_level(convertLogLevel(m_currentLevel));

    // Create new logger with existing sinks plus new console sink
    auto sinks = m_logger->sinks();
    sinks.push_back(consoleSink);

    auto newLogger =
      std::make_shared<spdlog::logger>(m_name, sinks.begin(), sinks.end());
    newLogger->set_level(convertLogLevel(m_currentLevel));
    newLogger->set_pattern("[%Y-%m-%d %H:%M:%S.%e] [%n] [%l] %v");

    // Replace the logger
    spdlog::drop(m_name);
    m_logger = newLogger;
    spdlog::register_logger(m_logger);
}

void SpdlogLogger::traceFormatted(const std::string& format, ...)
{
    if (!m_logger || !shouldLog(LogLevel::Trace))
        return;

    va_list args;
    va_start(args, format);
    std::string message = formatMessage(format, args);
    va_end(args);

    m_logger->trace(message);
}

void SpdlogLogger::debugFormatted(const std::string& format, ...)
{
    if (!m_logger || !shouldLog(LogLevel::Debug))
        return;

    va_list args;
    va_start(args, format);
    std::string message = formatMessage(format, args);
    va_end(args);

    m_logger->debug(message);
}

void SpdlogLogger::infoFormatted(const std::string& format, ...)
{
    if (!m_logger || !shouldLog(LogLevel::Info))
        return;

    va_list args;
    va_start(args, format);
    std::string message = formatMessage(format, args);
    va_end(args);

    m_logger->info(message);
}

void SpdlogLogger::warnFormatted(const std::string& format, ...)
{
    if (!m_logger || !shouldLog(LogLevel::Warning))
        return;

    va_list args;
    va_start(args, format);
    std::string message = formatMessage(format, args);
    va_end(args);

    m_logger->warn(message);
}

void SpdlogLogger::errorFormatted(const std::string& format, ...)
{
    if (!m_logger || !shouldLog(LogLevel::Error))
        return;

    va_list args;
    va_start(args, format);
    std::string message = formatMessage(format, args);
    va_end(args);

    m_logger->error(message);
}

void SpdlogLogger::criticalFormatted(const std::string& format, ...)
{
    if (!m_logger || !shouldLog(LogLevel::Critical))
        return;

    va_list args;
    va_start(args, format);
    std::string message = formatMessage(format, args);
    va_end(args);

    m_logger->critical(message);
}

spdlog::level::level_enum SpdlogLogger::convertLogLevel(LogLevel level) const
{
    switch (level)
    {
    case LogLevel::Trace:
        return spdlog::level::trace;
    case LogLevel::Debug:
        return spdlog::level::debug;
    case LogLevel::Info:
        return spdlog::level::info;
    case LogLevel::Warning:
        return spdlog::level::warn;
    case LogLevel::Error:
        return spdlog::level::err;
    case LogLevel::Critical:
        return spdlog::level::critical;
    case LogLevel::Off:
        return spdlog::level::off;
    default:
        return spdlog::level::info;
    }
}

LogLevel SpdlogLogger::convertSpdlogLevel(spdlog::level::level_enum level) const
{
    switch (level)
    {
    case spdlog::level::trace:
        return LogLevel::Trace;
    case spdlog::level::debug:
        return LogLevel::Debug;
    case spdlog::level::info:
        return LogLevel::Info;
    case spdlog::level::warn:
        return LogLevel::Warning;
    case spdlog::level::err:
        return LogLevel::Error;
    case spdlog::level::critical:
        return LogLevel::Critical;
    case spdlog::level::off:
        return LogLevel::Off;
    default:
        return LogLevel::Info;
    }
}

std::string SpdlogLogger::formatMessage(
  const std::string& format, va_list args
) const
{
    // Calculate required buffer size
    va_list argsCopy;
    va_copy(argsCopy, args);
    int size = std::vsnprintf(nullptr, 0, format.c_str(), argsCopy);
    va_end(argsCopy);

    if (size <= 0)
    {
        return format; // Return original format string if formatting fails
    }

    // Format the message
    std::string result(size + 1, '\0');
    std::vsnprintf(&result[0], result.size(), format.c_str(), args);
    result.resize(size); // Remove null terminator

    return result;
}

} // namespace logging
