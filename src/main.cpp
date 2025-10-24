// Copyright (c) 2025 JackLee
//
// This software is released under the MIT License.
// https://opensource.org/licenses/MIT

#include <QApplication>
#include <QCommandLineParser>
#include <opencv2/opencv.hpp>
#include "logging/Logging.h"
#include "logging/LoggerManager.h"
#include "faceDetectionWindow.h"

int main(int argc, char* argv[])
{
    QApplication app(argc, argv);
    // TODO: Version control may be handled via CMake or other build systems
    // later
    app.setApplicationVersion("1.0.0");

    // Setup command line parser
    QCommandLineParser parser;
    parser.setApplicationDescription(
      "Human face detection application using OpenCV and Qt5"
    );
    parser.addHelpOption();
    parser.addVersionOption();

    // Add log level option
    QCommandLineOption logLevelOption(
      QStringList() << "l" << "log-level",
      "Set the logging level (debug, info, warning, error, critical)",
      "level",
      "info"
    );
    parser.addOption(logLevelOption);

    // Add legacy debug option for backward compatibility
    QCommandLineOption debugOption(
      QStringList() << "d" << "debug",
      "Enable debug logging (equivalent to --log-level debug)"
    );
    parser.addOption(debugOption);

    // Add quiet option
    QCommandLineOption quietOption(
      QStringList() << "q" << "quiet",
      "Suppress all console output except errors"
    );
    parser.addOption(quietOption);

    // Add verbose option
    QCommandLineOption verboseOption(
      QStringList() << "verbose",
      "Enable verbose output (equivalent to --log-level debug)"
    );
    parser.addOption(verboseOption);

    // Parse the command line arguments
    parser.process(app);

    // Initialize logging system with settings-based configuration
    logging::LoggerManager::Config loggingConfig;

    // Parse command line arguments for logging configuration
    logging::LogLevel cmdLineLogLevel = logging::LogLevel::Info;
    bool logLevelOverridden = false;

    // Handle quiet mode (takes highest precedence)
    if (parser.isSet(quietOption))
    {
        cmdLineLogLevel = logging::LogLevel::Error;
        logLevelOverridden = true;
    }
    // Check if debug flag is set
    else if (parser.isSet(debugOption) || parser.isSet(verboseOption))
    {
        cmdLineLogLevel = logging::LogLevel::Debug;
        logLevelOverridden = true;
    }
    // Otherwise check log level option
    else if (parser.isSet(logLevelOption))
    {
        QString levelStr = parser.value(logLevelOption).toLower();
        cmdLineLogLevel = logging::stringToLogLevel(levelStr.toStdString());
        logLevelOverridden = true;
    }

    // Initialize the logging system
    logging::LoggerManager::getInstance().initialize(loggingConfig);

    int result = 0;
    {
        // Create and show the face detection window in a scope
        // This ensures the window is destroyed before logging shutdown
        FaceDetectionWindow window;
        window.show();

        result = app.exec();
    } // Window destructor called here

    // Explicitly shutdown logging system before application ends
    // This prevents static destruction order issues

    logging::LoggerManager::getInstance().shutdown();

    return result;
}
