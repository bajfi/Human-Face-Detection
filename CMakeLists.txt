cmake_minimum_required(VERSION 3.12)
project(HFaceDetector)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTORCC ON)
set(CMAKE_AUTOUIC ON)

# set release build by default
set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)

# Prefer system Qt over other Qt lib
set(CMAKE_PREFIX_PATH "/usr/lib/x86_64-linux-gnu/cmake/Qt5" ${CMAKE_PREFIX_PATH})

# Find Qt
find_package(Qt5 COMPONENTS Widgets Core Gui Charts REQUIRED)

# Find OpenCV - include DNN module for modern face detection
find_package(OpenCV REQUIRED COMPONENTS opencv_core opencv_imgproc opencv_objdetect opencv_imgcodecs opencv_videoio opencv_highgui opencv_dnn)

# Check if OpenCV has CUDA support
if(OpenCV_FOUND)
    # Try to find CUDA components in OpenCV
    find_package(OpenCV QUIET COMPONENTS opencv_cudaimgproc opencv_cudaarithm)
    if(OpenCV_opencv_cudaimgproc_FOUND AND OpenCV_opencv_cudaarithm_FOUND)
        message(STATUS "OpenCV CUDA support found")
        set(OPENCV_HAS_CUDA TRUE)
        add_definitions(-DOPENCV_DNN_CUDA)
    else()
        message(STATUS "OpenCV CUDA support not found")
        set(OPENCV_HAS_CUDA FALSE)
    endif()
endif()

# Find spdlog
find_package(spdlog REQUIRED)

# Include directories
include_directories(${Qt5Widgets_INCLUDE_DIRS})
include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(${spdlog_INCLUDE_DIRS})

# Source files
add_executable(${PROJECT_NAME}
    src/main.cpp
    src/faceDetectionWindow.cpp
    src/faceDetectionWindow.h
    src/faceDetectionWindow.ui
    src/IFaceDetector.h
    src/CascadeFaceDetector.cpp
    src/CascadeFaceDetector.h
    src/YunetFaceDetector.cpp
    src/YunetFaceDetector.h
    src/FaceDetectorFactory.cpp
    src/FaceDetectorFactory.h
    src/ModelValidator.cpp
    src/ModelValidator.h
    # Video processing system
    src/VideoFaceDetector.cpp
    src/VideoFaceDetector.h
    src/PerformanceMonitor.cpp
    src/PerformanceMonitor.h
    src/AdaptiveMemoryPool.cpp
    src/AdaptiveMemoryPool.h
    src/ConfigurationManager.cpp
    src/ConfigurationManager.h
    # Logging system
    src/logging/LoggerManager.cpp
    src/logging/LoggerManager.h
    src/logging/SpdlogLogger.cpp
    src/logging/SpdlogLogger.h
    src/logging/ILogger.h
    src/logging/Logging.h
)

# Link libraries - include DNN module
target_link_libraries(${PROJECT_NAME}
    Qt5::Widgets Qt5::Core Qt5::Gui Qt5::Charts
    opencv_core opencv_imgproc opencv_objdetect opencv_imgcodecs opencv_videoio opencv_highgui opencv_dnn
    spdlog::spdlog
)

# Link CUDA libraries if available
if(OPENCV_HAS_CUDA)
    target_link_libraries(${PROJECT_NAME} opencv_cudaimgproc opencv_cudaarithm)
    message(STATUS "Linking with OpenCV CUDA libraries")
endif()


# Build tests when BUILD_TESTING is ON
option(BUILD_TESTING "Build the testing tree." ON)

# Test suites
if(BUILD_TESTING)
    enable_testing()
    add_subdirectory(tests)
endif()
