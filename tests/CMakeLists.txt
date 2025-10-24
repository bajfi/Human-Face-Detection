# Face Detection Tests Configuration

# Use GTest framework
find_package(GTest REQUIRED)
include(GoogleTest)

# Include parent directories for source files
set(FACE_DETECTION_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../src)

# Test executable for Cascade Face Detector
add_executable(test_cascade_detector
    test_cascade_detector.cpp
    ${FACE_DETECTION_DIR}/CascadeFaceDetector.cpp
)

target_include_directories(test_cascade_detector PRIVATE
    ${FACE_DETECTION_DIR}
    ${OpenCV_INCLUDE_DIRS}
)

# Link CUDA libraries if available
if(OPENCV_HAS_CUDA)
    target_link_libraries(test_cascade_detector opencv_cudaimgproc opencv_cudaarithm)
    target_compile_definitions(test_cascade_detector PRIVATE -DOPENCV_DNN_CUDA)
endif()

target_link_libraries(test_cascade_detector
    GTest::gtest
    GTest::gtest_main
    opencv_core
    opencv_imgproc
    opencv_objdetect
    opencv_imgcodecs
    opencv_highgui
    opencv_dnn
)


# Test executable for DNN Face Detector
add_executable(test_dnn_detector
    test_dnn_detector.cpp
    ${FACE_DETECTION_DIR}/YunetFaceDetector.cpp
)

target_include_directories(test_dnn_detector PRIVATE
    ${FACE_DETECTION_DIR}
    ${OpenCV_INCLUDE_DIRS}
)

target_link_libraries(test_dnn_detector
    GTest::gtest
    GTest::gtest_main
    opencv_core
    opencv_imgproc
    opencv_objdetect
    opencv_imgcodecs
    opencv_highgui
    opencv_dnn
)

# Test executable for Integration Tests
add_executable(test_integration
    test_integration.cpp
    ${FACE_DETECTION_DIR}/CascadeFaceDetector.cpp
    ${FACE_DETECTION_DIR}/YunetFaceDetector.cpp
    ${FACE_DETECTION_DIR}/FaceDetectorFactory.cpp
)

target_include_directories(test_integration PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${FACE_DETECTION_DIR}
    ${OpenCV_INCLUDE_DIRS}
)

target_link_libraries(test_integration
    GTest::gtest
    GTest::gtest_main
    opencv_core
    opencv_imgproc
    opencv_objdetect
    opencv_imgcodecs
    opencv_highgui
    opencv_dnn
)

# Link CUDA libraries if available
if(OPENCV_HAS_CUDA)
    target_link_libraries(test_integration opencv_cudaimgproc opencv_cudaarithm)
    target_compile_definitions(test_integration PRIVATE -DOPENCV_DNN_CUDA)
endif()

# Discover and add tests to CTest
gtest_discover_tests(test_cascade_detector)
gtest_discover_tests(test_dnn_detector)
gtest_discover_tests(test_integration)

# Copy test resources to build directory
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/test_images/test01.jpg
               ${CMAKE_CURRENT_BINARY_DIR}/test_images/test01.jpg COPYONLY)

configure_file(${CMAKE_CURRENT_SOURCE_DIR}/../models/haarcascade_frontalface_alt.xml
               ${CMAKE_CURRENT_BINARY_DIR}/models/haarcascade_frontalface_alt.xml COPYONLY)

configure_file(${CMAKE_CURRENT_SOURCE_DIR}/../models/face_detection_yunet_2023mar.onnx
               ${CMAKE_CURRENT_BINARY_DIR}/models/face_detection_yunet_2023mar.onnx COPYONLY)
