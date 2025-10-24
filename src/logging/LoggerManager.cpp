#include "LoggerManager.h"
#include <filesystem>

namespace logging
{

LoggerManager& LoggerManager::getInstance()
{
    static LoggerManager instance;
    return instance;
}

LoggerManager::~LoggerManager()
{
    shutdown();
}

void LoggerManager::initialize(const Config& config)
{
    std::lock_guard<std::mutex> lock(m_mutex);

    if (m_initialized)
    {
        return; // Already initialized
    }

    m_config = config;
    m_globalLevel = config.defaultLevel;

    // Create log directory if file logging is enabled
    if (config.enableFileLogging && !config.logDirectory.empty())
    {
        try
        {
            std::filesystem::create_directories(config.logDirectory);
        }
        catch (const std::exception&)
        {
            // Continue without file logging if directory creation fails
        }
    }

    // Create default logger
    createDefaultLogger();

    m_initialized = true;
}

void LoggerManager::initialize()
{
    Config defaultConfig;
    initialize(defaultConfig);
}

LoggerPtr LoggerManager::getLogger()
{
    std::lock_guard<std::mutex> lock(m_mutex);

    // Don't auto-initialize if we've been explicitly shut down
    if (m_shutdown)
    {
        return nullptr;
    }

    if (!m_initialized)
    {
        // Don't auto-initialize - require explicit initialization
        // This prevents premature initialization with default (Info) level
        // before main.cpp can set the correct level from command line
        return nullptr;
    }

    return m_defaultLogger;
}

LoggerPtr LoggerManager::getLogger(const std::string& name)
{
    std::lock_guard<std::mutex> lock(m_mutex);

    // Don't auto-initialize if we've been explicitly shut down
    if (m_shutdown)
    {
        return nullptr;
    }

    if (!m_initialized)
    {
        // Don't auto-initialize - require explicit initialization
        // This prevents premature initialization with default (Info) level
        // before main.cpp can set the correct level from command line
        return nullptr;
    }

    // Check if logger already exists
    if (m_loggers.contains(name))
    {
        return m_loggers[name];
    }

    // Create new logger with default configuration
    SpdlogLogger::Config loggerConfig;
    loggerConfig.name = name;
    loggerConfig.level = m_globalLevel;
    loggerConfig.pattern = m_config.defaultPattern;
    loggerConfig.enableConsole = m_config.enableConsoleLogging;
    loggerConfig.enableFile = m_config.enableFileLogging;

    if (m_config.enableFileLogging)
    {
        loggerConfig.filename = generateLogFilename(name);
        loggerConfig.maxFileSize = m_config.maxFileSize;
        loggerConfig.maxFiles = m_config.maxFiles;
    }

    auto logger = std::make_shared<SpdlogLogger>(loggerConfig);
    m_loggers[name] = logger;

    return logger;
}

LoggerPtr LoggerManager::createLogger(
  const std::string& name, const SpdlogLogger::Config& config
)
{
    std::lock_guard<std::mutex> lock(m_mutex);

    // Don't create new loggers if we've been explicitly shut down
    if (m_shutdown)
    {
        return nullptr;
    }

    if (!m_initialized)
    {
        // Don't auto-initialize - require explicit initialization
        // This prevents premature initialization with default (Info) level
        // before main.cpp can set the correct level from command line
        return nullptr;
    }

    auto logger = std::make_shared<SpdlogLogger>(config);
    m_loggers[name] = logger;

    // Create default logger if it doesn't exist
    if (!m_defaultLogger)
    {
        createDefaultLogger();
    }

    return logger;
}

void LoggerManager::setGlobalLevel(LogLevel level)
{
    std::lock_guard<std::mutex> lock(m_mutex);

    m_globalLevel = level;

    // Update all existing loggers
    for (auto& [name, logger] : m_loggers)
    {
        logger->setLevel(level);
    }

    if (m_defaultLogger)
    {
        m_defaultLogger->setLevel(level);
    }
}

LogLevel LoggerManager::getGlobalLevel() const
{
    std::lock_guard<std::mutex> lock(m_mutex);
    return m_globalLevel;
}

void LoggerManager::flushAll()
{
    std::lock_guard<std::mutex> lock(m_mutex);

    for (auto& [name, logger] : m_loggers)
    {
        logger->flush();
    }

    if (m_defaultLogger)
    {
        m_defaultLogger->flush();
    }

    // Also flush spdlog's global registry
    spdlog::apply_all(
      [](std::shared_ptr<spdlog::logger> l)
      {
          l->flush();
      }
    );
}

void LoggerManager::shutdown()
{
    std::lock_guard<std::mutex> lock(m_mutex);

    if (!m_initialized || m_shutdown)
    {
        return;
    }

    // Flush all loggers before shutdown
    for (auto& [name, logger] : m_loggers)
    {
        if (logger)
        {
            logger->flush();
        }
    }

    if (m_defaultLogger)
    {
        m_defaultLogger->flush();
    }

    // Clear all loggers
    m_loggers.clear();
    m_defaultLogger.reset();

    // Shutdown spdlog

    spdlog::shutdown();

    m_initialized = false;
    m_shutdown = true;
}

void LoggerManager::createDefaultLogger()
{
    if (m_defaultLogger)
    {
        return; // Already created
    }

    SpdlogLogger::Config loggerConfig;
    loggerConfig.name = DEFAULT_LOGGER_NAME;
    loggerConfig.level = m_globalLevel;
    loggerConfig.pattern = m_config.defaultPattern;
    loggerConfig.enableConsole = m_config.enableConsoleLogging;
    loggerConfig.enableFile = m_config.enableFileLogging;

    if (m_config.enableFileLogging)
    {
        if (!m_config.defaultLogFile.empty())
        {
            loggerConfig.filename = m_config.defaultLogFile;
        }
        else
        {
            loggerConfig.filename = generateLogFilename(DEFAULT_LOGGER_NAME);
        }
        loggerConfig.maxFileSize = m_config.maxFileSize;
        loggerConfig.maxFiles = m_config.maxFiles;
    }

    m_defaultLogger = std::make_shared<SpdlogLogger>(loggerConfig);
    m_loggers[DEFAULT_LOGGER_NAME] = m_defaultLogger;
}

std::string LoggerManager::generateLogFilename(
  const std::string& loggerName
) const
{
    if (m_config.logDirectory.empty())
    {
        return loggerName + ".log";
    }

    std::filesystem::path logPath(m_config.logDirectory);
    logPath /= (loggerName + ".log");
    return logPath.string();
}

} // namespace logging
